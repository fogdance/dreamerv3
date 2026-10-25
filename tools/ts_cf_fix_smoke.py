"""DEV_ONLY: four real frozen-pi2 forks, one complete Native comparison.

Inputs are read-only. --out must be a new diagnostic directory, outside the
formal run. This script performs inference/replay, never training. Run with
the sibling Gym src and this Dreamer checkout on PYTHONPATH.
"""
import argparse
import copy
import json
import pickle
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from dreamerv3.thinking_student import feedback as F, value as V
from dreamerv3.thinking_student.run import Autonomous
from gym_trading_env.research import ts_v1_cf as CF, ts_v1_corpus as C
from gym_trading_env.research.ts_v1_account import StudentEnv


def read_pairs(folder):
    return {p['cursor']: p for p in map(json.loads, (folder / 'pairs.jsonl').open())}


def full_t7(policy, corpus, cf, out):
    """Disable both coupling routes; run every step of the actual frozen pi2."""
    work = out / 'full_t7'
    work.mkdir()
    began = time.monotonic()
    with V._PeakMemory() as memory:
        V._cf_init(str(policy), str(corpus), str(work))
        with (cf / 'work' / 'reference.pkl').open('rb') as f:
            reference = pickle.load(f)['reference']
        with (cf / 'work' / '7.pkl').open('rb') as f:
            saved = pickle.load(f)
        assert saved['phase'] == 'fork' and not saved['progress'].rewards
        reference.continuations, reference.floor = {}, np.inf
        run = Autonomous.restore(V._CF_WORKER['agent'], V._CF_WORKER['env'], saved['run'])
        stepper = V.Stepper(run)
        result = CF.run_branch(stepper, reference, 0, max_steps=reference.n)
        result.update(native_steps=result['steps'], policy_calls=stepper.policy_calls,
                      native_status=run.env.status, native_terminal=bool(run.env.terminated),
                      final_equity=float(run.env._calculate_equity()))
    result.update(seconds=time.monotonic() - began, peak_rss_bytes=memory.peak)
    assert result['reason'] == 'TERMINAL' and result['native_terminal']
    exact = read_pairs(cf)[7]
    result['exact_g'], result['q_error'] = exact['g'], exact['g'] - result['g']
    assert abs(result['q_error']) < 1e-12
    V._json_write(out / 'full_t7.json', result)
    print('FULL_NATIVE_T7', json.dumps(result), flush=True)
    return result


def funding_boundary(corpus, evidence, out):
    """Artificial cash is diagnostic only, never added to CF teaching pairs."""
    with (evidence / 'CZCE.FG201_31153_native.pkl').open('rb') as f:
        original = pickle.load(f)
    env = StudentEnv(C.corpus_env(corpus, market=False))
    reference_obs, _ = env.reset(seed=0, options={'resume': original})
    original_state = CF.native_continuation(env)
    with (out / 'CZCE.FG201' / 'work' / 'reference.pkl').open('rb') as f:
        reference = pickle.load(f)['reference']
    cursor = int(env.current_step)
    continuation = reference.continuations[cursor]
    assert CF.same_state(original_state, continuation['native'])
    env.step(int(reference.action[reference.row(cursor)]))
    reference_lots = float(env.net_lots)
    results = []
    for capital in ('11018.99', '11019.01', '11475.01'):
        snapshot = copy.deepcopy(original)
        gap = Decimal(capital) - Decimal(snapshot['balances']['user_cash'])
        snapshot['balances']['user_cash'] = capital
        snapshot['balances']['broker_pnl'] = str(Decimal(snapshot['balances']['broker_pnl']) - gap)
        obs, _ = env.reset(seed=0, options={'resume': snapshot})
        native = CF.native_continuation(env)
        equal = CF.same_state(original_state, native)
        state = dict(cursor=cursor, equity=float(env._calculate_equity()), native=native,
                     policy=continuation['policy'], **{k: obs[k] for k in CF.INPUTS})
        kind = reference.coupling_kind(state)
        assert kind is None
        _, reward, terminated, truncated, info = env.step(1)
        results.append(dict(artificial=True, policy_trajectory=False, capital=capital,
            exact_equal=bool(equal), coupling_kind=kind, mask=obs['action_mask'].tolist(),
            first_joint_after_fork=cursor + 1 if reference_lots == 0 and not env.net_lots else None,
            input_differences={k: int(np.count_nonzero(obs[k] != reference_obs[k])) for k in CF.INPUTS},
            funds_safe=bool(reference.future_min_equity[reference.row(cursor)] + min(
                float(Decimal(capital) - Decimal(original['balances']['user_cash'])), 0.) >= reference.floor),
            account_native=obs['account_native'].tolist(), reward=reward,
            terminated=terminated, truncated=truncated, status=info['status'],
            reject=info.get('reject_reason'), equity=float(env._calculate_equity())))
    assert not any(r['exact_equal'] for r in results)
    assert results[0]['reject'] == 'MASKED_ADD' and not results[0]['terminated']
    assert results[1]['status'] == 'MARGIN_CALL' and results[1]['terminated']
    assert results[2]['status'] == 'ACTIVE' and not results[2]['terminated']
    V._json_write(out / 'funding_boundary.json', results)
    return results


def existing_labels(run_root, corpus_root, out):
    """Read the old FG201 labels and exercise the real merge/read interfaces.
    No old branch is simulated; only the two measured new records are overlaid.
    """
    name = 'CZCE.FG201'
    policy = run_root / 'policies' / 'pi_0002'
    collection = run_root / 'experience' / 'pi_0002' / 'collect' / name
    base = run_root / 'experience' / 'pi_0002' / 'cf' / name
    with (out / name / 'work' / 'reference.pkl').open('rb') as f:
        reference = pickle.load(f)['reference']
    rows = V._base_pairs(base, policy, corpus_root / name, collection, reference)
    old_valid = {t: r for t, r in rows.items() if r['status'] == 'VALID_COMPLETE'}
    assert len(old_valid) == 250
    new = read_pairs(out / name)
    assert not (old_valid.keys() & new.keys())  # both real probes previously were CAP
    rows.update(new)
    assert all(rows[t] == r for t, r in old_valid.items())
    combined = out / 'supplement_compatibility'
    combined.mkdir(exist_ok=True)
    (combined / 'pairs.jsonl').write_text(''.join(json.dumps(rows[t]) + '\n' for t in sorted(rows)))
    targets = V.cf_targets(reference, list(rows.values()))
    np.savez(combined / 'targets.npz', **targets)
    old_meta = json.loads((base / 'meta.json').read_text())
    meta = {k: old_meta[k] for k in ('policy', 'corpus', 'collection', 'gamma', 'clock',
                                    'reference_rows', 'legal_flat_decisions', 'forks', 'labels')}
    meta['valid_pairs'] = int(targets['q_cf_valid'].sum())
    meta['counts'] = {}
    for row in rows.values():
        key = f"{row['status']}/{row['reason']}"
        meta['counts'][key] = meta['counts'].get(key, 0) + 1
    meta.update(kind='DEV_ONLY_READER_COMPATIBILITY', simulated_branches=0)
    V._json_write(combined / 'meta.json', meta)
    reader = V.QWindows(collection, combined / 'targets.npz', 'train', holdout=0)
    assert int(reader.data['q_cf_valid'].sum()) == 251
    feedback = combined / 'feedback'
    F.main(['--collection', str(collection), '--corpus', str(corpus_root / name),
            '--cf', str(combined), '--out', str(feedback), '--kappa-floor', '0.05'])
    items, summary = F.load(feedback)
    assert int(items['cf_valid'].sum()) == 251 and not items['cf_valid'][20789]
    result = dict(old_valid_retained=250, new_valid=1, combined_valid=251,
                  q_reader_valid=251, s5_valid=summary['sources']['cf']['items'], native_steps=0)
    V._json_write(out / 'existing_labels.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--corpus-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy = args.run / 'policies' / 'pi_0002'
    collect = args.run / 'experience' / 'pi_0002' / 'collect'
    report = dict(kind='DEV_ONLY_SMALL_REAL_NATIVE', policy=str(policy), contracts={})
    for contract, cursors in [('CZCE.FG201', [7, 20789]), ('CZCE.FG205', [243, 3932])]:
        corpus, cf = args.corpus_root / contract, args.out / contract
        probe = V.run_cf(policy, corpus, collect / contract, cf, cursors=cursors, workers=1, budget=0)
        V._json_write(args.out / f'{contract}_probe.json', probe)
        probed = read_pairs(cf)
        short, long = (probed[t] for t in cursors)
        assert short['status'] == 'VALID_COMPLETE' and short['completed_in'] == 'probe'
        assert long['status'] == 'UNAVAILABLE_BUDGET' and long['steps'] == 64 and long['g'] is None
        if contract == 'CZCE.FG201':
            assert (short['coupling_kind'], short['steps'], short['coupled_at']) == ('EXACT_STATE', 4, 11)
            assert not short['first_joint']['funds_safe'] and short['first_joint']['native_equal']
        final = V.run_cf(policy, corpus, collect / contract, cf, cursors=cursors, workers=1, budget=1)
        pairs = read_pairs(cf)
        assert final['compute']['reference']['reused']
        assert final['compute']['executed_native_steps'] == pairs[cursors[1]]['steps'] - 64
        if contract == 'CZCE.FG201':
            assert (pairs[20789]['reason'], pairs[20789]['steps']) == ('CAP', 8192)
            assert not pairs[20789]['native_terminal']
            assert pairs[20789]['first_joint']['equity_gap'] == 564.
            assert not pairs[20789]['first_joint']['native_equal']
        else:
            assert (pairs[243]['steps'], pairs[3932]['steps']) == (1, 418)
            assert all(p['coupling_kind'] == 'FUNDS_SAFE' for p in pairs.values())
        reader = V.QWindows(collect / contract, cf / 'targets.npz', 'train', holdout=0)
        assert int(reader.data['q_cf_valid'].sum()) == final['valid_pairs']
        feedback = args.out / 'feedback' / contract
        F.main(['--collection', str(collect / contract), '--corpus', str(corpus),
                '--cf', str(cf), '--out', str(feedback), '--kappa-floor', '0.05'])
        items, summary = F.load(feedback)
        assert int(items['cf_valid'].sum()) == final['valid_pairs']
        assert not any(items['cf_valid'][p['cursor']] for p in pairs.values() if p['g'] is None)
        old = read_pairs(args.run / 'experience' / 'pi_0002' / 'cf' / contract)
        report['contracts'][contract] = dict(probe=probe, final=final, before={t: old[t] for t in cursors},
            after=pairs, feedback_sources=summary['sources'],
            prior_individual_wall_seconds='not measured in this run')
        V._json_write(args.out / 'results.json', report)
    report['full_t7'] = full_t7(policy, args.corpus_root / 'CZCE.FG201', args.out / 'CZCE.FG201', args.out)
    report['funding_boundary'] = funding_boundary(args.corpus_root / 'CZCE.FG201', args.evidence, args.out)
    report['existing_labels'] = existing_labels(args.run, args.corpus_root, args.out)
    V._json_write(args.out / 'results.json', report)
    print('PASS: four forks, resumable quotas, full Native Q, funding boundaries, Q Reader and S5', flush=True)


if __name__ == '__main__':
    main()
