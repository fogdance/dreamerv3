"""Thinking Student V1 runner: staged training on TS teaching material, frozen
policy snapshots and autonomous Native experience.

  python -m dreamerv3.thinking_student.run train-market  --run DIR --corpus DIR [--corpus DIR ...]
  python -m dreamerv3.thinking_student.run train-outcome --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run train-policy  --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run freeze  --run DIR --name pi0 --corpus DIR ... [--out DIR]
  python -m dreamerv3.thinking_student.run collect --policy DIR --corpus DIR --out DIR
  python -m dreamerv3.thinking_student.run check   --out DIR --corpus DIR
  python -m dreamerv3.thinking_student.run rollout --run DIR --corpus DIR
  python -m dreamerv3.thinking_student.run cf      --policy DIR --corpus DIR [--forks N]
  python -m dreamerv3.thinking_student.run state-check --policy DIR --collection DIR
  python -m dreamerv3.thinking_student.run train-value --run DIR --init-from POLICY --cf DIR [--cf DIR ...]
  python -m dreamerv3.thinking_student.run freeze  --run DIR --name pi1 --beta-q 1 --kappa-q auto --cf DIR ...
  python -m dreamerv3.thinking_student.run market-carry (--run DIR | --policy DIR) --corpus DIR ... --out ROOT

One policy improvement round (S4) runs in one directory: train-value from
the frozen pi_k on its CF targets, train-policy to refresh the prior
(L_policy_demo with a neutral anchor + L_reason), then freeze pi_{k+1}.

S5 (L6): --feedback names the teaching items of pi_k (feedback.py, one
folder per contract). train-outcome and train-policy then draw a share
--correct-fraction of every batch around its correct answers and a share
--remedial-fraction around its errors: outcome answers (the L2 review), or
decisions whose better action the graded teacher demonstrates (L5);
train-value samples its correct and wrong decision rows --correct-boost and
--remedial-boost times more. The items are OFFLINE_TRAIN material
(TIME_ORDERED_NOT_SUPPORTED), and freeze lists them in the lineage.

Windows of L2-L5 start from the zero market state with a loss burn-in, or
with --window-start carry from the causal market carry of the continuous
path, stored once per contract by market-carry under --carry-root (the
market brain is frozen after L1, so one store serves every later stage).

--corpus is a contract folder of gym ts_v1_corpus (teaching material and
the TS env inputs: Quality TP reward, no CHUNK_END). The training commands
take several (C4: windows of several TRAIN contracts, never joined across
contracts); collect, check, rollout and cf take one. --holdout 0 trains on
every TRAIN window; a holdout > 0 is a TRAIN_INTERNAL_DIAGNOSTIC tail of
each contract, never a DEV exam.

A run directory holds `ckpt/<step>/agent.pkl` with `progress.json` (the
updates of each stage so far) and a `latest` pointer; --save-every also
saves during a training command, so an interrupted command loses at most
the updates since its last save. Loading never checks code or data hashes
(C2); shape mismatches fail loudly instead of silently re-initializing. A
frozen policy `policies/<name>/` holds the complete parameters and
`policy.json`, the record of everything that decides its actions (A6 2.4);
collection and exams load it and never train it, and a frozen folder is
never written again.
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from . import labels as L
from .config import observation_space
from .corpus import (MIXES, LABEL_ACCOUNT, MultiCorpusWindows, TSCorpus, corpus_sources,
                     pop_carry)
from .student import make_agent

POLICY_OBS = tuple(observation_space())
DECISION_OBS = tuple(k for k in POLICY_OBS
                     if k not in ('reward', 'is_first', 'is_last', 'is_terminal'))


# ---- checkpoint ----

STAGES = ('L1', 'L2', 'L3', 'L4')


def _write(path, data):
  """Replace a file atomically: a reader sees the old or the new bytes."""
  tmp = path.with_name(path.name + '.tmp')
  tmp.write_bytes(data)
  tmp.rename(path)


def save(agent, run, tag=None, progress=None):
  """`ckpt/<updates>/agent.pkl` (with the stage `progress`), then the
  `latest` pointer."""
  run = Path(run)
  data = agent.save()
  name = tag or f"{data['counters']['updates']:012d}"
  folder = run / 'ckpt' / name
  folder.mkdir(parents=True, exist_ok=True)
  _write(folder / 'agent.pkl', pickle.dumps(data))
  if progress is not None:
    _write(folder / 'progress.json', json.dumps(progress, indent=1).encode())
  _write(run / 'ckpt' / 'latest', name.encode())
  return folder


def latest(run):
  pointer = Path(run) / 'ckpt' / 'latest'
  return Path(run) / 'ckpt' / pointer.read_text().strip() if pointer.exists() else None


def resume(agent, run):
  folder = latest(run)
  if folder is None:
    return None
  with (folder / 'agent.pkl').open('rb') as f:
    agent.load(pickle.load(f))
  return folder


def progress(run):
  """Optimizer updates of each stage in this run so far, as recorded with
  its latest checkpoint (zero before the first recorded save)."""
  done = dict.fromkeys(STAGES, 0)
  folder = latest(run)
  if folder is not None and (folder / 'progress.json').exists():
    done.update(json.loads((folder / 'progress.json').read_text())['updates'])
  return done


REPORTS = {'L1': 'train_market', 'L2': 'train_outcome', 'L3': 'train_policy',
           'L4': 'train_value'}


def reports(run, names=tuple(REPORTS.values())):
  """The newest report of each training command among the run's
  checkpoints: one improvement round trains L4 and then refreshes L3."""
  found, ckpt = {}, Path(run) / 'ckpt'
  folders = sorted(p for p in ckpt.iterdir() if p.is_dir()) if ckpt.exists() else []
  for folder in folders:
    for name in names:
      if (folder / f'{name}.json').exists():
        found[name] = json.loads((folder / f'{name}.json').read_text())
  return found


def build(run, preset, device, init_f1=None, init_from=None, record=True, **kwargs):
  """Resume the run strictly; a fresh run may start from F1 market weights
  or from another Thinking Student checkpoint (`init_from`, listed changes),
  listed in inherit.json unless record=False (a reader such as
  market-carry)."""
  if init_f1 and init_from:
    raise ValueError('Choose one of init_f1 and init_from')
  agent = make_agent(Path(run) / 'logdir', preset=preset, device=device, **kwargs)
  folder = resume(agent, run)
  lists = None
  if folder is not None:
    print(f'Resumed {folder}')
  elif init_from:
    from .transfer import inherit_student
    lists = inherit_student(agent, init_from)
  elif init_f1:
    from .transfer import inherit_f1
    lists = inherit_f1(agent, init_f1)
  if lists is not None and record:
    Path(run).mkdir(parents=True, exist_ok=True)
    (Path(run) / 'inherit.json').write_text(json.dumps(lists, indent=1))
  return agent


# ---- frozen complete policy snapshot (pi_k) ----

def _flat(tree, prefix):
  out = {}
  for k, v in tree.items():
    if isinstance(v, dict):
      out.update(_flat(v, f'{prefix}{k}.'))
    else:
      out[f'{prefix}{k}'] = list(v) if isinstance(v, tuple) else v
  return out


def _repos():
  """Non-blocking provenance of the two code bases (never a load gate)."""
  out = {}
  root = Path(__file__).resolve().parents[2]
  for name, path in (('dreamerv3', root), ('gym', root.parent / 'gym-trading-env')):
    try:
      sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
      dirty = bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True))
      out[name] = dict(path=str(path), sha=sha, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
      out[name] = None
  return out


def freeze(agent, run, name, preset, corpora=(), teaching=None, lineage=None, out=None):
  """Write the snapshot `out` (default `run/policies/<name>/`): the complete
  parameters (every module that can change an action) and policy.json with
  the action selection, decision coefficients, RSSM initialization, Reward,
  the TRAIN contracts it learned from and its lineage. A snapshot never
  changes: a folder that holds a policy.json is refused, and a new policy
  takes a new name (C4 4.3-4)."""
  folder = Path(out) if out else Path(run) / 'policies' / name
  if (folder / 'policy.json').exists():
    raise FileExistsError(f'{folder} holds a frozen policy; a new policy takes a new name')
  sources = corpus_sources(corpora)
  folder.mkdir(parents=True, exist_ok=True)
  data = agent.save()
  blob = pickle.dumps(data)
  meta = dict(
      policy=name, created=time.strftime('%Y-%m-%d %H:%M:%S'),
      source_checkpoint=str(latest(run)), updates=int(data['counters']['updates']),
      params_sha256=hashlib.sha256(pickle.dumps(
          {k: np.asarray(v) for k, v in sorted(data['params'].items())
           if not k.startswith('opt_')})).hexdigest(),
      action_selection='DETERMINISTIC_MASKED_ARGMAX',
      action_rule='ADD iff the Native mask allows ADD and z > 0; z == 0 -> WAIT',
      behavior_policy=dict(mode='explore', kind='epsilon mixture around the formal action',
                           eps=float(agent.config.ts.explore_eps)),
      rssm_initialization=(
          'zero market carry (rssm.initial) once, at the first market minute: the '
          'TS env has no CHUNK_END, so market memory runs to the true terminal; '
          'training windows start from the zero carry with a loss burn-in, or, '
          "where a teaching report says window_start 'carry', from the stored "
          'causal carry of the continuous path; stochastic latents use the policy '
          'call counter seeds'),
      outcome=dict(horizon_minutes=L.HORIZON, bins=list(L.OUTCOME_BINS),
                   classes=[*(f'{e}_bin{k}' for e in L.EVENTS for k in range(L.NBINS)),
                            'OPEN_AT_H'],
                   answers='p_tp_h, p_sl_h, p_other_h, p_open_h: first ending within H; '
                           'mfe_15, mfe_60, mfe_120: largest executable favourable move '
                           '(points) within that many market minutes of the fill'),
      reasoning=dict(reason_pref=(
          'logit of the Reasoning answer "the graded teacher demonstrates ADD"; '
          'L_reason in L3 runs trains it (teaching.train_policy, loss/reason); '
          'a diagnostic, never part of z')),
      preset=preset, ts=_flat(dict(agent.config.ts), ''),
      # z near 0 can change sign between compute dtypes; collection, CF and
      # exams load the policy on the device and dtype recorded here.
      inference=dict(platform=agent.config.jax.platform,
                     compute_dtype=agent.config.jax.compute_dtype),
      teaching=teaching or {}, lineage=lineage or {}, repos=_repos(),
      corpora=[dict(path=str(c), reward_schema=s['inputs']['reward_schema'],
                    **{k: s[k] for k in ('contract', 'fold', 'mode', 'funds_floor',
                                         'gamma', 'clock', 'grades', 'created')})
               for c, s in zip(corpora, sources)])
  if sources:
    # corpus_sources admits only corpora of one reward schema.
    meta['reward_version'] = sources[0]['inputs']['reward_schema']
  _write(folder / 'agent.pkl', blob)
  _write(folder / 'policy.json', json.dumps(meta, indent=1).encode())
  return folder


def load_policy(folder, device=None, logdir=None, log=print, ts=None):
  """Build the student exactly as the snapshot recorded it (by default on
  its recorded inference device and dtype) and load it. The snapshot's
  non-optimizer parameters fix its actions and must all match; optimizer
  state of a training stage added after the freeze never acts and starts
  fresh (listed). `ts` overrides decision settings of the snapshot for a
  frozen-weight diagnostic (E1: decision.beta_q); the weights are unchanged."""
  folder = Path(folder)
  meta = json.loads((folder / 'policy.json').read_text())
  updates = {f'agent.ts.{k}': tuple(v) if isinstance(v, list) else v
             for k, v in {**meta['ts'], **(ts or {})}.items()}
  agent = make_agent(logdir or folder / 'logdir', preset=meta['preset'],
                     device=device or meta['inference']['platform'],
                     batch_size=1, batch_length=1, updates=updates)
  with (folder / 'agent.pkl').open('rb') as f:
    data = pickle.load(f)
  fresh = agent.save()['params']
  missing = sorted(fresh.keys() - data['params'].keys())
  extra = sorted(data['params'].keys() - fresh.keys())
  if extra or any(not k.startswith('opt_') for k in missing):
    raise ValueError(f'{folder}: snapshot parameters differ from the student '
                     f'(missing {missing[:5]}, extra {extra[:5]})')
  if missing:
    log(f'{folder.name}: fresh optimizer state for {len(missing)} keys of later stages')
    data['params'] = {**{k: fresh[k] for k in missing}, **data['params']}
  agent.load(data)
  return agent, meta


# ---- autonomous Native experience through the single student.policy ----

ANSWER_KEYS = ('p_add', 'z', 'q_wait', 'q_add', 'delta_q', 'alpha', 'anchor',
               'correction', 'reason_pref', 'outcome_valid', 'p_tp_h', 'p_tp_15',
               'p_sl_h', 'p_other_h', 'p_open_h', 'quality', *L.MFE_HORIZONS)
ACCOUNT_KEYS = ('tp_count', 'non_tp_exits', 'account_net_pnl', 'unfinished_lots')


def student_obs(obs, reward, first, last, terminal):
  out = {k: np.asarray(obs[k])[None] for k in DECISION_OBS}
  out.update(reward=np.float32([reward]), is_first=np.array([first]),
             is_last=np.array([last]), is_terminal=np.array([terminal]))
  return out


class Autonomous:
  """The student trades its own Native account through student.policy.

  mode='policy' executes the formal pi_k; mode='explore' executes the
  behavior policy mu and records mu(a|s). Each step returns the answer sheet
  recorded before the action, the proposed and executed actions and the
  Native account experience. Teachers never act here. Market memory starts
  once, at the first market minute, and runs to the true terminal (the TS
  env has no CHUNK_END). `state()` holds the Native account, market memory,
  pending observation and the policy seed counter, so `restore()` continues
  exactly where the run stopped.
  """

  def __init__(self, agent, env, mode='policy', seed=0):
    if env.chunk_steps:
      raise ValueError('The Thinking Student trades the TS env without CHUNK_END (chunk_steps 0)')
    self.agent, self.env, self.mode = agent, env, mode
    self.obs, _ = env.reset(seed=seed)
    self.carry = agent.init_policy(1)
    self.reward, self.first, self.steps, self.done = 0., True, 0, False

  def step(self, action=None):
    """One decision through student.policy. A given `action` is executed in
    place of the policy's (a counterfactual branch); the policy still runs,
    so market memory and policy seeds advance as on the policy's own path."""
    env, obs = self.env, self.obs
    o = student_obs(obs, self.reward, self.first, False, False)
    self.carry, act, out = self.agent.policy(self.carry, o, mode=self.mode)
    forced = action is not None
    action = int(action) if forced else int(act['action'][0])
    mask = obs['action_mask'].astype(float).tolist()
    if action == 1 and mask[1] < 0.5:
      raise RuntimeError('Student policy executed a masked ADD')
    cursor = int(env.current_step)
    clock = str(env._raw.index[cursor])
    equity = float(env._calculate_equity())
    nxt, reward, terminated, truncated, info = env.step(action)
    row = dict(step=self.steps, cursor=cursor, time=clock, mode=self.mode,
               first=self.first, mask=mask, proposed=int(out['proposed'][0]),
               executed=action, forced=forced, mu_prob=float(out['mu_prob'][0]),
               explored=bool(out['explored'][0]), reward=float(reward),
               equity=equity, status=info['status'],
               reject_reason=info.get('reject_reason'),
               **{k: float(out[f'answer/{k}'][0]) for k in ANSWER_KEYS},
               **{k: info.get(k) for k in ACCOUNT_KEYS})
    # Decision-time arrays; `reward` is what this action produced in Native.
    arrays = {k: np.asarray(obs[k]) for k in DECISION_OBS}
    arrays.update(
        is_first=self.first, action=action, proposed=row['proposed'],
        mu_prob=row['mu_prob'], explored=row['explored'], reward=row['reward'],
        terminal=bool(terminated), **{f'answer_{k}': row[k] for k in ANSWER_KEYS})
    self.steps += 1
    self.reward, self.first = float(reward), False
    self.done = bool(terminated or truncated)
    self.obs = nxt
    return row, arrays

  def state(self):
    import jax
    return dict(env=self.env.export_state(), obs=self.obs, reward=self.reward,
                first=self.first, steps=self.steps, done=self.done, mode=self.mode,
                carry=jax.tree.map(np.asarray, self.carry),
                n_actions=int(self.agent.n_actions))

  @classmethod
  def restore(cls, agent, env, state):
    self = cls.__new__(cls)
    self.agent, self.env, self.mode = agent, env, state['mode']
    obs, _ = env.reset(seed=0, options={'resume': state['env']})
    for key, value in state['obs'].items():
      if not np.array_equal(obs[key], value):
        raise ValueError(f'Resumed Native observation {key} differs from the saved one')
    self.obs, self.carry = obs, state['carry']
    self.reward, self.first = state['reward'], state['first']
    self.steps, self.done = state['steps'], state['done']
    agent.n_actions.value = state['n_actions']
    return self


def collect(agent, env, out, steps, mode='policy', chunk=1024, policy=None,
            log=print):
  """Autonomous experience until `steps` decisions in total (or the data end):
  steps/<n>.jsonl answer and account records, replay/<n>.npz decision arrays,
  then state.pkl, so a stopped collection resumes at its last full chunk.
  snapshots/<n>.pkl keeps the state after each chunk (a hard link of that
  state.pkl): the CF replays the reference path from them in parallel."""
  out = Path(out)
  for sub in ('steps', 'replay', 'snapshots'):
    (out / sub).mkdir(parents=True, exist_ok=True)
  file = out / 'state.pkl'
  if file.exists():
    with file.open('rb') as f:
      saved = pickle.load(f)
    run, index, totals = Autonomous.restore(agent, env, saved['run']), saved['chunks'], saved['totals']
    totals['resumed'] += 1
    log(f'Resumed collection at step {run.steps}, chunk {index}')
  else:
    run, index = Autonomous(agent, env, mode), 0
    totals = dict(steps=0, legal=0, adds=0, explored=0, reward=0., resumed=0)
  meta = dict(policy=policy, mode=run.mode, eps=float(agent.config.ts.explore_eps)
              if run.mode == 'explore' else 0., chunk=chunk,
              platform=agent.config.jax.platform,
              compute_dtype=agent.config.jax.compute_dtype)
  while run.steps < steps and not run.done:
    rows, arrays = [], []
    while len(rows) < chunk and run.steps < steps and not run.done:
      row, arr = run.step()
      rows.append(row)
      arrays.append(arr)
    np.savez(out / 'replay' / f'{index:06d}.npz',
             **{k: np.stack([a[k] for a in arrays]) for k in arrays[0]})
    with (out / 'steps' / f'{index:06d}.jsonl').open('w') as f:
      f.writelines(json.dumps(r) + '\n' for r in rows)
    totals['steps'] += len(rows)
    totals['legal'] += sum(r['mask'][1] > 0.5 for r in rows)
    totals['adds'] += sum(r['executed'] == 1 for r in rows)
    totals['explored'] += sum(r['explored'] for r in rows)
    totals['reward'] += sum(r['reward'] for r in rows)
    index += 1
    tmp = file.with_suffix('.tmp')
    with tmp.open('wb') as f:
      pickle.dump(dict(run=run.state(), chunks=index, totals=totals), f)
    tmp.rename(file)
    snapshot = out / 'snapshots' / f'{index:06d}.pkl'
    if not snapshot.exists():
      os.link(file, snapshot)
    last = rows[-1]
    summary = dict(
        **meta, **totals, chunks=index, done=run.done, last_time=last['time'],
        add_ratio=totals['adds'] / max(totals['legal'], 1),
        tp_count=last['tp_count'], non_tp_exits=last['non_tp_exits'],
        account_net_pnl=last['account_net_pnl'])
    (out / 'meta.json').write_text(json.dumps(summary, indent=1))
    log(f'chunk {index}: steps={totals["steps"]} adds={totals["adds"]} '
        f'legal={totals["legal"]} tp={last["tp_count"]} '
        f'non_tp={last["non_tp_exits"]} reward={totals["reward"]:.3f}')
  return run


FACT_KINDS = {'TAKE_PROFIT': 'TP', 'STOP_LOSS': 'SL'}


def check_collection(out, corpus=None, inputs=None):
  """Minimal verification of an autonomous collection (C3 S2): each recorded
  proposal follows the formal rule, the formal pi executed its own proposal
  (forced counterfactual actions excepted), no executed ADD was masked, and
  each closed Native trade reached TP by >= the TP distance or lost >= the SL
  distance at SL, the distances of the env inputs the collection ran on
  (`inputs`, by default the corpus's inputs/env.yaml; under entry.stop_rule
  the trade's own, recomputed from the market). With the corpus of
  the collection's contract, each student ADD from a funded flat account is
  compared with the Native fact of the same cursor."""
  from gym_trading_env.research import ts_v1_corpus as C
  out = Path(out)
  if inputs is None and corpus is None:
    raise ValueError('check_collection needs the corpus or the env inputs of the collection')
  native = C.native_env(inputs or Path(corpus) / 'inputs' / 'env.yaml', market=False)
  tp_points, sl_points = native.take_profit_points, native.stop_loss_points
  rows = [json.loads(line) for file in sorted((out / 'steps').glob('*.jsonl'))
          for line in file.open()]
  with (out / 'state.pkl').open('rb') as f:
    env = pickle.load(f)['run']['env']
  legal = sum(r['mask'][1] > 0.5 for r in rows)
  adds = [r for r in rows if r['executed'] == 1]
  report = dict(
      rows=len(rows), legal=legal, adds=len(adds), add_ratio=len(adds) / max(legal, 1),
      explored=sum(r['explored'] for r in rows),
      rule_violations=[r['step'] for r in rows
                       if r['proposed'] != int(r['mask'][1] > 0.5 and r['z'] > 0)],
      formal_violations=[r['step'] for r in rows if r['mode'] == 'policy' and not r['forced']
                         and (r['executed'] != r['proposed'] or r['mu_prob'] != 1.)],
      masked_adds=[r['step'] for r in adds if r['mask'][1] < 0.5])
  trades, exits, points = env['attributions'], {}, []
  for t in trades:
    exits[t['reason']] = exits.get(t['reason'], 0) + 1
    move = (1 if t['side'] == 'LONG' else -1) * (
        Decimal(t['exit_price']) - Decimal(t['entry_price']))
    if native.stop_rule:
      plan = native.planned_exit(t['entry_cursor'] - 1, t['entry_price'], t['side'])
      # The trade's own distances, either side of its fill (SHORT or LONG).
      tp_points = abs(Decimal(t['entry_price']) - plan['take_profit_price'])
      sl_points = abs(plan['stop_loss_price'] - Decimal(t['entry_price']))
    if (t['reason'] == 'TAKE_PROFIT' and move < tp_points) or (
        t['reason'] == 'STOP_LOSS' and move > -sl_points):
      points.append(dict(entry_cursor=t['entry_cursor'], reason=t['reason'], move=str(move)))
  report.update(closed=len(trades), exits=exits, open=len(env['entries']),
                point_violations=points, reward=sum(r['reward'] for r in rows),
                last={k: rows[-1][k] for k in ('time', 'cursor', *ACCOUNT_KEYS)})
  if corpus:
    report['facts'] = _fact_agreement(rows, adds, trades, env['entries'], corpus)
  return report


def _fact_agreement(rows, adds, trades, entries, corpus):
  """Fill, exit type and exit minute of each student ADD against the fact of
  its cursor; a trade still open at the collection end agrees when the fact
  exits later."""
  from gym_trading_env.research import ts_v1_corpus as C
  facts = C.load_arrays(Path(corpus) / 'facts')
  floor = json.loads((Path(corpus) / 'source.json').read_text())['funds_floor']
  closed = {t['entry_cursor']: t for t in trades}
  still_open = {e['entry_cursor'] for e in entries.values()}
  result = dict(agree=0, rejected=0, unchecked=0, disagree=[])
  for r in adds:
    t = r['cursor']
    filled = not r['reject_reason']
    if r['equity'] < floor:
      result['unchecked'] += 1
    elif filled != bool(facts['filled'][t]):
      result['disagree'].append(dict(cursor=t, native_filled=filled, fact_filled=not filled))
    elif not filled:
      result['rejected'] += 1
    elif int(facts['entry_cursor'][t]) in closed:
      trade = closed[int(facts['entry_cursor'][t])]
      kind, exit = C.KINDS[facts['kind'][t]], int(facts['exit_cursor'][t])
      if FACT_KINDS.get(trade['reason'], 'OTHER_EXIT') == kind and trade['exit_cursor'] == exit:
        result['agree'] += 1
      else:
        result['disagree'].append(dict(cursor=t, native=trade['reason'],
            native_exit=trade['exit_cursor'], fact=kind, fact_exit=exit))
    elif int(facts['entry_cursor'][t]) in still_open and (
        facts['exit_cursor'][t] < 0 or facts['exit_cursor'][t] > rows[-1]['cursor']):
      result['agree'] += 1
    else:
      result['disagree'].append(dict(cursor=t, native='NO_TRADE',
          fact=C.KINDS[facts['kind'][t]], fact_exit=int(facts['exit_cursor'][t])))
  return result


def rollout(agent, env, steps, mode='policy', record=None):
  """A short autonomous run kept in memory (engineering smoke and tests)."""
  run, rows = Autonomous(agent, env, mode), []
  while len(rows) < steps and not run.done:
    row, _ = run.step()
    rows.append(row)
    if record is not None:
      record.write(json.dumps(row) + '\n')
  return run, rows


# ---- the causal market carry of a contract (training windows that inherit it) ----

def market_carry(agent, corpus, out, source, counter=None, log=print):
  """out/<contract>/: the market carry the student holds before each cursor
  of the corpus contract on the continuous path from the first market minute
  (corpus.MarketCarry). The market inputs and policy seeds are those of a
  formal collection, so for every policy with this market brain and seed
  counter the store is its deployed market memory; the account inputs (of
  LABEL_ACCOUNT) never reach the market brain. Each contract starts from the
  zero state and from the policy seed counter `counter` (default: the
  agent's), as each collection starts from a fresh load of the frozen
  policy, never from the memory or seeds of another contract. Training
  windows that start from it continue that memory instead of the zero state
  (user decision 2026-10-24 after C4 review R3)."""
  from .corpus import LABEL_ACCOUNT, MarketCarry
  began, n = time.time(), corpus.rows
  data = corpus.window(LABEL_ACCOUNT, 0, n)
  base = int(agent.n_actions) if counter is None else int(counter)
  agent.n_actions.value = base
  carry, arrays = agent.init_policy(1), {}
  for t in range(n):
    for k in ('deter', 'stoch'):
      value = np.asarray(carry['market'][k][0])
      arrays.setdefault(k, np.zeros((n, *value.shape), value.dtype))[t] = value
    if t < n - 1:
      obs = student_obs({k: data[k][t] for k in DECISION_OBS}, data['reward'][t], t == 0,
                        False, False)
      carry, _, _ = agent.policy(carry, obs, mode='policy')
  folder = Path(out) / corpus.source['contract']
  MarketCarry.write(folder, arrays, dict(
      contract=corpus.source['contract'], corpus=str(corpus.folder), rows=n, source=source,
      seed_counter=base, platform=agent.config.jax.platform,
      compute_dtype=agent.config.jax.compute_dtype, seconds=round(time.time() - began, 1),
      created=time.strftime('%Y-%m-%d %H:%M:%S')))
  log(f'market carry of {corpus.source["contract"]}: {n} cursors in {time.time() - began:.0f} s')
  return folder


# ---- staged training on natural corpus sequences ----

LOG_KEYS = {
    'L1': ('opt_l1/loss', 'loss/dyn', 'loss/market_seq_e_shared',
           'loss/account_can_add', 'brain/p_add', 'brain/delta_q_abs'),
    'L2': ('opt_l2/loss', 'loss/outcome_event', 'loss/outcome_risk',
           'loss/outcome_mae', 'loss/outcome_mfe', 'loss/outcome_quality', 'outcome/labels',
           'outcome/p_tp_on_tp', 'outcome/p_tp_on_sl', 'outcome/p_other_h'),
    'L3': ('opt_l3/loss', 'loss/policy_demo', 'loss/reason', 'policy/n_add',
           'policy/p_add_on_add', 'policy/p_add_on_wait', 'policy/add_rate',
           'policy/formal_add_rate', 'reason/acc_add', 'reason/acc_wait',
           'brain/alpha'),
    'L4': ('opt_l4/loss', 'loss/q_real', 'loss/q_cf', 'loss/q_pair', 'q/n_cf',
           'q/delta_q_on_cf', 'q/delta_g', 'q/sign_agree', 'brain/delta_q_abs'),
}


def _device(agent, data, counter):
  from embodied.jax import internal
  data = internal.device_put(data, agent.train_sharded)
  data['seed'] = agent._seeds(counter, agent.train_mirrored)
  return data


def _start(agent, data, init):
  """The market state the windows of `data` start from: the stored causal
  carry they bring ('carry/*', removed from data), else init(B), the zero
  state."""
  carry = pop_carry(data)
  return agent.restore_carry({'market': carry}) if carry else init(len(data['is_first']))


def train_stage(agent, windows, updates, log_every=10, on_update=None):
  """`updates` optimizer steps of the configured A6 stage on corpus windows."""
  stage, mets = agent.config.ts.stage, {}
  for i in range(updates):
    data = windows.sample(agent.config.batch_size, agent.config.batch_length)
    carry = _start(agent, data, agent.init_train)
    data = _device(agent, data, int(agent.n_batches))
    agent.n_batches.increment()
    _, _, mets = agent.train(carry, data)
    if (i + 1) % log_every == 0 or i == updates - 1:
      print(f'update {int(agent.n_updates)}', {
          k: round(float(mets[k]), 4) for k in LOG_KEYS[stage] if k in mets})
    if on_update:
      on_update(i + 1)
  return mets


# Per stage: the scored metric prefixes, the default denominator and the
# metrics pooled over another count. Counts themselves are summed.
POOL = {
    'L2': (('loss/', 'outcome/'), 'outcome/labels', {
        'loss/outcome_risk': 'outcome/n_risk', 'loss/outcome_mae': 'outcome/n_mae',
        'loss/outcome_mfe': 'outcome/n_mfe', 'loss/outcome_quality': 'outcome/n_quality',
        'outcome/p_tp_on_tp': 'outcome/n_tp', 'outcome/p_tp_on_sl': 'outcome/n_sl'}),
    'L3': (('loss/', 'policy/', 'reason/'), 'policy/n_demo', {
        'policy/p_add_on_add': 'policy/n_add', 'policy/acc_add': 'policy/n_add',
        'policy/p_add_on_wait': 'policy/n_wait', 'policy/acc_wait': 'policy/n_wait',
        'policy/add_rate': 'policy/n_legal', 'policy/formal_add_rate': 'policy/n_legal',
        'reason/p_add_on_add': 'policy/n_add', 'reason/acc_add': 'policy/n_add',
        'reason/p_add_on_wait': 'policy/n_wait', 'reason/acc_wait': 'policy/n_wait'}),
    'L4': (('loss/', 'q/', 'brain/'), 'q/n_real', {
        'loss/q_cf': 'q/n_cf', 'loss/q_pair': 'q/n_cf', 'q/delta_g': 'q/n_cf',
        'q/delta_q_on_cf': 'q/n_cf', 'q/sign_agree': 'q/n_sign'}),
}
COUNTS = ('outcome/labels', 'outcome/label_not_legal', 'policy/n_demo',
          'policy/n_add', 'policy/n_wait', 'policy/n_legal', 'q/n_real', 'q/n_cf',
          'q/n_sign')


def evaluate(agent, windows, constant=None):
  """Stage report pooled over every held-out window, with the best
  state-free answer scored on exactly the same rows: `constant` (a class
  distribution) for L2, the uniform WAIT/ADD answer for class-weighted L3,
  the mean train targets (`constant`, QWindows.constant) for L4."""
  from .student import warm
  stage, ts = agent.config.ts.stage, agent.config.ts
  prefixes, default, pools = POOL[stage]
  B, T = agent.config.batch_size, agent.config.batch_length
  sums, weights, batches = {}, {}, 0

  def pool(key, value, weight):
    sums[key] = sums.get(key, 0.) + weight * float(value)
    weights[key] = weights.get(key, 0.) + weight

  for i, data in enumerate(windows.batches(B, T)):
    if stage == 'L2' and constant is not None:
      w = np.asarray(warm(data['is_first'], ts.outcome_burnin))
      w = w * data['outcome_valid']
      nll = -np.log((data['outcome_allowed'] * constant).sum(-1) + (w == 0))
      pool('constant/outcome_event', (nll * w).sum() / max(w.sum(), 1.), w.sum())
    if stage == 'L3':
      # The uniform answer scores log 2 on both class-weighted terms.
      w = np.asarray(warm(data['is_first'], ts.demo_burnin)) * data['demo_weight']
      n = float((w > 0).sum())
      for key in ('policy_demo', 'reason'):
        pool(f'constant/{key}', np.log(2.) * w.sum() / max(n, 1.), n)
    if stage == 'L4' and constant is not None:
      w = np.asarray(warm(data['is_first'], ts.q_burnin))
      real, cf = w * data['q_real_valid'], w * data['q_cf_valid']
      dg = np.where(data['action'] == 1, data['q_real'] - data['q_cf'],
                    data['q_cf'] - data['q_real'])
      for key, err, mask in (
          ('q_real', data['q_real'] - constant['q_real'], real),
          ('q_cf', data['q_cf'] - constant['q_cf'], cf),
          ('q_pair', dg - constant['delta_g'], cf)):
        pool(f'constant/{key}', (err ** 2 * mask).sum() / max(mask.sum(), 1.), mask.sum())
      # The state-free sign answer: every pair gets the sign of the train mean.
      signed = cf * (dg != 0)
      agree = signed * (np.sign(dg) == np.sign(constant['delta_g']))
      pool('constant/sign_agree', agree.sum() / max(signed.sum(), 1.), signed.sum())
    carry = _start(agent, data, agent.init_report)
    _, mets = agent.report(carry, _device(agent, data, i))
    batches += 1
    for k, v in mets.items():
      if not k.startswith(prefixes) or (
          k.startswith(('outcome/n_', 'policy/n_', 'q/n_')) and k not in COUNTS):
        continue
      if k in COUNTS:
        sums[k] = sums.get(k, 0.) + float(v)
      else:
        pool(k, v, float(mets[pools.get(k, default)]))
  out = {k: round(v / max(weights[k], 1.), 5) if k in weights else int(v)
         for k, v in sorted(sums.items())}
  return {'updates': int(agent.n_updates), 'batches': batches, **out}


def constant_answer(train):
  """Add-one smoothed class frequencies of the observed train outcomes: the
  best Outcome answer that ignores the state, as a reference to beat."""
  fit = train.outcome_allowed()
  observed = fit[fit.sum(-1) == 1]
  return (observed.sum(0) + 1.) / (len(observed) + fit.shape[-1])


def corpus_env(corpus):
  """The Native TS env of the corpus contract from a fresh account, with the
  student's account input."""
  from gym_trading_env.research import ts_v1_corpus as C
  from gym_trading_env.research.ts_v1_account import StudentEnv
  return StudentEnv(C.corpus_env(corpus))


def _train(agent, args, train, held, report, constant=None):
  """`args.updates` updates of the run's stage on the `train` windows: a
  checkpoint with the stage progress every `save_every` updates and at the
  end; with held-out windows (TRAIN_INTERNAL_DIAGNOSTIC) an evaluation
  before, every `eval_every` updates and after."""
  stage, before = agent.config.ts.stage, progress(args.run)

  def record(done):
    return dict(updates={**before, stage: before[stage] + done}, stage=stage,
                agent_updates=int(agent.n_updates), seed=args.seed, mix=args.mix,
                sources=report.get('corpora') or report.get('contracts'))

  def evaluate_now():
    report['eval'].append(evaluate(agent, held, constant))
    print('eval', report['eval'][-1])

  def on_update(done):
    if done == args.updates:
      return
    if held is not None and args.eval_every and done % args.eval_every == 0:
      evaluate_now()
    if args.save_every and done % args.save_every == 0:
      print('Saved', save(agent, args.run, progress=record(done)))

  if held is not None:
    evaluate_now()
  mets = train_stage(agent, train, args.updates, args.log_every, on_update)
  if held is not None:
    evaluate_now()
  report.update(last_train={k: float(v) for k, v in mets.items() if k.startswith(LOG_KEYS[stage])},
                exposure=train.exposure, curriculum=train.curriculum(),
                progress=record(args.updates))
  return save(agent, args.run, progress=report['progress'])


def value_anchor(args):
  """beta_Q and kappa_Q of a new snapshot (A5). Q moves an action only where
  it learned an action difference: beta_Q > 0 needs at least two valid
  ADD/WAIT pairs in the --cf train targets of one frozen pi_k, else the
  snapshot keeps beta_Q = 0 and says Q_ANCHOR_NOT_READY. kappa 'auto' is
  the std of Delta G over those pairs, pooled over their contracts with the
  --mix weights; --kappa-floor only replaces a zero or smaller std of them
  (MultiQWindows.kappa)."""
  record = dict(beta_q=float(args.beta_q))
  if args.kappa_q != 'auto':
    record['kappa_q'] = float(args.kappa_q)
    if not record['kappa_q'] > 0:
      raise ValueError(f'kappa_Q must be positive, got {record["kappa_q"]}')
    if not record['beta_q'] > 0:
      return record
  if not args.cf:
    raise ValueError("beta_Q > 0 and kappa_Q 'auto' need the --cf targets of the value head")
  from .value import MultiQWindows
  targets = MultiQWindows(args.cf, 'train', holdout=args.holdout, mix=args.mix)
  record['pairs'] = targets.pair_counts()
  if record['pairs']['total'] < 2:
    print(f'Q_ANCHOR_NOT_READY: {record["pairs"]["total"]} valid ADD/WAIT pairs; beta_Q stays 0')
    return dict(record, beta_q=0., requested_beta_q=record['beta_q'], kappa_q=None,
                status='Q_ANCHOR_NOT_READY', cf=[str(c) for c in args.cf])
  if args.kappa_q == 'auto':
    kappa, source = targets.kappa(args.kappa_floor)
    record.update(kappa_q=kappa, kappa_source=dict(
        cf=[str(c) for c in args.cf], split='train', holdout=args.holdout, **source))
  return record


def kept_anchor(policy):
  """The value anchor of a frozen policy for a snapshot whose round skipped
  L4: the value brain is the one inherited from it, so are beta_Q and
  kappa_Q."""
  data = json.loads((Path(policy) / 'policy.json').read_text())
  return dict(beta_q=float(data['ts']['decision.beta_q']),
              kappa_q=float(data['ts']['decision.kappa_q']), status='L4_SKIPPED_ANCHOR_KEPT',
              kept_from=str(policy), kept=data['teaching'].get('value_anchor'))


def _feedback(args, kind):
  """({'remedial': {contract: error cursors} or None, 'correct': ...} of the
  --feedback folders, their report) for one kind (S5 B2); a set whose
  fraction is 0 (boost 1 for train-value) passes no cursors. Without
  feedback ({}, None)."""
  if not args.feedback:
    return {}, None
  from . import feedback as F
  summaries = F.summaries(args.feedback)
  found = F.feedback_sets(args.feedback, kind)
  if kind == 'decision':
    on = dict(remedial=args.remedial_boost != 1., correct=args.correct_boost != 1.)
  else:
    on = dict(remedial=args.remedial_fraction > 0, correct=args.correct_fraction > 0)
  sets = {name: found[name] if on[name] else None for name in F.FEEDBACK_SETS}
  return sets, dict(
      folders=[str(f) for f in args.feedback], kind=kind, policy=summaries[0]['policy'],
      cf=[s['cf'] for s in summaries], on=on,
      cursors={name: {c: len(v) for c, v in found[name].items()} for name in F.FEEDBACK_SETS},
      remedial_fraction=args.remedial_fraction, remedial_boost=args.remedial_boost,
      correct_fraction=args.correct_fraction, correct_boost=args.correct_boost,
      teaching_order=F.OFFLINE_TRAIN, time_ordered=F.TIME_ORDERED_NOT_SUPPORTED,
      note='OFFLINE_TRAIN teaching items, not a strict time-ordered online learning result')


def _one(folders, flag):
  if not folders or len(folders) != 1:
    raise ValueError(f'This command takes exactly one {flag}')
  return folders[0]


def student_updates(args):
  """ts.market and ts.outcome_points of a new student from --market-macd /
  --context-15m / --outcome-points; a frozen policy keeps the ones recorded
  in its snapshot."""
  return {**({} if args.market_macd is None else {'agent.ts.market.macd': bool(args.market_macd)}),
          **({} if args.context_15m is None else {'agent.ts.market.context_15m': args.context_15m}),
          **({} if args.outcome_points is None else {'agent.ts.outcome_points': args.outcome_points})}


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('command', choices=[
      'train-market', 'train-outcome', 'train-policy', 'freeze', 'collect',
      'check', 'rollout', 'cf', 'state-check', 'train-value', 'market-carry'])
  parser.add_argument('--run', default=None)
  parser.add_argument('--preset', default='size12m')
  parser.add_argument('--device', default=None,
                      help="cpu or cuda; default cuda (collect: the snapshot's device)")
  parser.add_argument('--market-macd', type=int, choices=(0, 1), default=None,
                      help='ts.market.macd of a new student (default student.yaml)')
  parser.add_argument('--context-15m', type=int, default=None,
                      help='ts.market.context_15m of a new student: complete 15m buckets')
  parser.add_argument('--outcome-points', type=float, default=None,
                      help='ts.outcome_points of a new student: points per MAE / MFE unit '
                           '(the corpora\'s, labels.outcome_points)')
  parser.add_argument('--init-f1', default=None)
  parser.add_argument('--init-from', default=None,
                      help='earlier Thinking Student run or checkpoint (listed inheritance)')
  parser.add_argument('--corpus', action='append', default=None,
                      help='TS corpus folder of one contract; training and freeze take several')
  parser.add_argument('--mix', default='uniform_contract', choices=MIXES,
                      help='how windows draw their contract')
  parser.add_argument('--holdout', type=int, default=2048,
                      help='last cursors of each contract held out of training '
                           '(TRAIN_INTERNAL_DIAGNOSTIC); 0 trains on every window')
  parser.add_argument('--updates', type=int, default=10)
  parser.add_argument('--save-every', type=int, default=0,
                      help='also checkpoint every N updates of a training command')
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--batch-length', type=int, default=64)
  parser.add_argument('--log-every', type=int, default=10)
  parser.add_argument('--eval-every', type=int, default=0)
  parser.add_argument('--demo-balance', type=float, default=1.0,
                      help='L3 class weighting: 1 equal class weight, 0 natural')
  parser.add_argument('--name', default='pi0', help='frozen policy name')
  parser.add_argument('--policy', default=None, help='frozen policy folder')
  parser.add_argument('--out', default=None,
                      help='collection folder (collect, check); snapshot folder (freeze); '
                           'report file (state-check); store root (market-carry)')
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--chunk', type=int, default=1024)
  parser.add_argument('--mode', default='policy', choices=['policy', 'explore'])
  parser.add_argument('--collection', default=None,
                      help='formal pi_k collection (cf; default collect/<policy>_policy)')
  parser.add_argument('--forks', type=int, default=None,
                      help='cf: WAIT forks sampled uniformly (default all); pi_k ADDs always')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--workers', type=int, default=4, help='cf: GPU worker processes')
  parser.add_argument('--max-steps', type=int, default=8192,
                      help='cf: branch steps before NOT_COMPARABLE')
  parser.add_argument('--budget', type=int, default=None,
                      help='cf: continuation quota after every candidate gets a short probe; '
                           'unfinished probes are sampled by decision-time strata (default: all)')
  parser.add_argument('--probe-steps', type=int, default=64,
                      help='cf: probe at most this many steps (1..64), inside --max-steps')
  parser.add_argument('--cursors', type=int, nargs='+', default=None,
                      help='cf: explicit legal fork cursors for a local supplement / small experiment')
  parser.add_argument('--base-cf', default=None,
                      help='cf: read-only existing pairs to supplement into a new --out')
  parser.add_argument('--value-anchor-of', default=None,
                      help="freeze: keep this frozen policy's beta_Q and kappa_Q (its round "
                           'skipped L4, the value brain is the inherited one)')
  parser.add_argument('--cf', action='append', default=None,
                      help='cf output folder of one contract (train-value, kappa); '
                           'several of one frozen pi_k')
  parser.add_argument('--windows', type=int, default=32,
                      help='state-check: windows compared with the continuous path')
  parser.add_argument('--window-start', default='zero', choices=['zero', 'carry'],
                      help='windows start from the zero market state (with the loss burn-in) '
                           'or from the causal market carry: train-outcome, train-policy and '
                           'train-value read it from --carry-root; state-check from '
                           '--carry-root (the store of its --corpus contract) or else from '
                           'the continuous path')
  parser.add_argument('--carry-root', default=None,
                      help='MarketCarry stores <root>/<contract>/ (market-carry writes them)')
  parser.add_argument('--seed-offset', type=int, default=0,
                      help='state-check: policy seeds of the windows shifted from the path')
  parser.add_argument('--cf-boost', type=float, default=16.,
                      help='train-value: sampling weight of a fork row with a valid branch')
  parser.add_argument('--beta-q', type=float, default=None,
                      help='freeze: A5 value-anchor weight beta_Q; above 0 it needs two valid '
                           '--cf pairs, else 0 (Q_ANCHOR_NOT_READY)')
  parser.add_argument('--kappa-q', default=None,
                      help="freeze: kappa_Q, a number or 'auto' = std(Delta G) of --cf train pairs")
  parser.add_argument('--kappa-floor', type=float, default=None,
                      help="freeze: positive kappa_Q for a zero or smaller 'auto' std of real "
                           'pairs; never for missing pairs')
  parser.add_argument('--feedback', action='append', default=None,
                      help='S5 teaching items of one contract (feedback.py) of the frozen pi_k: '
                           'correct and remedial windows in train-outcome, train-policy and '
                           'train-value; freeze lists them in the lineage')
  parser.add_argument('--correct-fraction', type=float, default=0.,
                      help='train-outcome, train-policy: share of every batch drawn around '
                           'correct cursors of --feedback')
  parser.add_argument('--remedial-fraction', type=float, default=0.,
                      help='train-outcome, train-policy: share of every batch drawn around '
                           'error cursors of --feedback')
  parser.add_argument('--correct-boost', type=float, default=1.,
                      help='train-value: sampling weight factor of a correct decision row')
  parser.add_argument('--remedial-boost', type=float, default=1.,
                      help='train-value: sampling weight factor of a decision-error row')
  args = parser.parse_args(argv)
  if args.feedback and args.command not in ('train-outcome', 'train-policy', 'train-value',
                                             'freeze'):
    parser.error(f'{args.command} takes no --feedback: L6 never trains the market brain')
  if not args.feedback and (args.remedial_fraction or args.remedial_boost != 1.
                            or args.correct_fraction or args.correct_boost != 1.):
    parser.error('--correct-fraction, --remedial-fraction, --correct-boost and --remedial-boost '
                 'need --feedback')
  if args.correct_fraction + args.remedial_fraction > 1:
    parser.error('--correct-fraction and --remedial-fraction together exceed the batch')
  if args.command == 'collect':
    agent, meta = load_policy(args.policy, args.device)
    out = args.out or Path(args.policy).parents[1] / 'collect' / f"{meta['policy']}_{args.mode}"
    print('Collecting into', out)
    collect(agent, corpus_env(_one(args.corpus, '--corpus')), out, args.steps, args.mode,
            args.chunk, meta['policy'])
    return
  if args.command == 'check':
    report = check_collection(args.out, _one(args.corpus, '--corpus'))
    (Path(args.out) / 'check.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return
  if args.command == 'cf':
    from .value import run_cf
    policy = Path(args.policy)
    name = json.loads((policy / 'policy.json').read_text())['policy']
    collection = args.collection or policy.parents[1] / 'collect' / f'{name}_policy'
    out = Path(args.out or policy.parents[1] / 'cf' / name)
    run_cf(policy, _one(args.corpus, '--corpus'), collection, out, args.forks, args.seed,
           args.workers, args.max_steps, budget=args.budget, probe_steps=args.probe_steps,
           cursors=args.cursors, base_cf=args.base_cf)
    return
  if args.command == 'state-check':
    from .value import window_check
    store = None
    if args.carry_root:
      if args.window_start != 'carry':
        parser.error('state-check reads --carry-root only with --window-start carry')
      store = Path(args.carry_root) / corpus_sources([_one(args.corpus, '--corpus')])[0]['contract']
    report = window_check(args.policy, args.collection, args.windows, args.batch_length,
                          args.seed, args.device, args.window_start, args.seed_offset, store)
    out = Path(args.out or Path(args.collection) / 'state_check.json')
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'windows_detail'}, indent=1))
    return
  if args.command == 'market-carry':
    if not args.corpus or not args.out:
      parser.error('market-carry needs --corpus and --out')
    if args.policy:
      agent, _ = load_policy(args.policy, args.device)
      source = dict(policy=str(Path(args.policy).resolve()))
    elif args.run:
      # The student as the run holds it (its latest checkpoint, else its
      # init source); nothing of the run is written.
      agent = build(args.run, args.preset, args.device or 'cuda', args.init_f1, args.init_from,
                    record=False, batch_size=1, batch_length=1, updates=student_updates(args))
      source = dict(run=str(Path(args.run).resolve()), checkpoint=str(latest(args.run) or ''),
                    init_from=args.init_from, init_f1=args.init_f1)
    else:
      parser.error('market-carry needs --policy or --run')
    counter = int(agent.n_actions)
    for folder in args.corpus:
      market_carry(agent, TSCorpus(folder, behaviors=[LABEL_ACCOUNT]), args.out, source, counter)
    return
  if not args.run:
    parser.error(f'{args.command} needs --run')
  stage = {'train-market': 'L1', 'train-outcome': 'L2', 'train-policy': 'L3',
           'train-value': 'L4'}.get(args.command, 'L1')
  carry_root = None
  if args.window_start == 'carry' and args.command in ('train-outcome', 'train-policy',
                                                       'train-value'):
    if not args.carry_root:
      parser.error('--window-start carry needs --carry-root')
    carry_root = args.carry_root
  elif args.window_start == 'carry' and args.command == 'train-market':
    parser.error('L1 trains the market brain: its windows cannot start from a stored carry')
  updates, anchor = {'agent.ts.stage': stage, **student_updates(args)}, None
  if args.command == 'freeze' and args.value_anchor_of:
    if args.beta_q is not None or args.kappa_q is not None or args.cf:
      parser.error('--value-anchor-of keeps an anchor: no --beta-q, --kappa-q or --cf')
    anchor = kept_anchor(args.value_anchor_of)
    updates['agent.ts.decision.beta_q'] = anchor['beta_q']
    updates['agent.ts.decision.kappa_q'] = anchor['kappa_q']
  elif args.command == 'freeze' and args.beta_q is not None:
    anchor = value_anchor(args)
    updates['agent.ts.decision.beta_q'] = anchor['beta_q']
    if anchor['kappa_q'] is not None:
      updates['agent.ts.decision.kappa_q'] = anchor['kappa_q']
  agent = build(args.run, args.preset, args.device or 'cuda', args.init_f1, args.init_from,
                batch_size=args.batch_size, batch_length=args.batch_length, updates=updates)
  ts = agent.config.ts
  # The sampler continues on a fresh stream after each save it resumes from.
  # The agent's update count, inherited from pi_k, separates the learners of
  # iteration rounds: each is a new run whose stage progress starts at 0, and
  # every round's L2 review and L5 would otherwise draw the same windows.
  seed = np.random.SeedSequence([args.seed, STAGES.index(stage), progress(args.run)[stage],
                                 int(agent.n_updates)])
  diagnostic = 'TRAIN_INTERNAL_DIAGNOSTIC' if args.holdout else None
  if args.command in ('train-market', 'train-outcome', 'train-policy'):
    sources = corpus_sources(args.corpus or ())
    if not sources:
      parser.error(f'{args.command} needs --corpus')
    sets, feedback = _feedback(args, 'outcome' if stage == 'L2' else 'demo')
    accounts = None if stage == 'L1' else [LABEL_ACCOUNT]
    corpora = [TSCorpus(c, behaviors=accounts, carry_root=carry_root) for c in args.corpus]
    for c in corpora:  # the MAE / MFE labels and the student's heads share one unit
      if c.outcome_points != float(ts.outcome_points):
        parser.error(f'{c.source["contract"]}: MAE / MFE labels in units of {c.outcome_points} '
                     f'points, the student\'s ts.outcome_points is {ts.outcome_points}')
    kw = dict(mix=args.mix, holdout=args.holdout, accounts=accounts)
    # Windows that start from the stored carry need no burn-in.
    if stage == 'L2':
      kw.update(weight='outcome', burnin=0 if carry_root else ts.outcome_burnin)
    elif stage == 'L3':
      kw.update(weight='demo', burnin=0 if carry_root else ts.demo_burnin)
    train = MultiCorpusWindows(corpora, split='train', seed=seed, demo_balance=args.demo_balance,
                               remedial=sets.get('remedial'),
                               remedial_fraction=args.remedial_fraction,
                               correct=sets.get('correct'), correct_fraction=args.correct_fraction,
                               **kw)
    held = MultiCorpusWindows(corpora, split='eval', demo_weights=train.demo_weights, **kw) \
        if args.holdout and stage != 'L1' else None
    report = dict(corpora=[s['contract'] for s in sources], corpus_folders=args.corpus,
                  mix=args.mix, holdout=args.holdout, diagnostic=diagnostic,
                  window_start='carry' if carry_root else 'zero', carry_root=carry_root,
                  rows={c: p.hi - p.lo for c, p in zip(train.contracts, train.parts)}, eval=[])
    if feedback:
      report['feedback'] = feedback
    constant = None
    if stage == 'L2':
      constant = constant_answer(train)
      report.update(train_labels=train.summary(), eval_labels=held and held.summary(),
                    constant_answer=constant.round(5).tolist(), mfe_minutes=list(L.MFE_HORIZONS),
                    grades=sources[0]['grades'])
    elif stage == 'L3':
      report.update(teacher=sources[0]['teacher'], grades=sources[0]['grades'],
                    demo_balance=args.demo_balance, demo_weights=train.demo_weights,
                    train_demos=train.demo_counts, eval_demos=held and held.demo_counts)
      print('demonstrations', {k: report[k] for k in ('demo_weights', 'train_demos')})
    folder = _train(agent, args, train, held, report, constant)
    (folder / f'{REPORTS[stage]}.json').write_text(json.dumps(report, indent=1))
    print('Saved', folder)
  elif args.command == 'train-value':
    from .value import MultiQWindows
    if not args.cf:
      parser.error('train-value needs --cf')
    kw = dict(holdout=args.holdout, burnin=0 if carry_root else ts.q_burnin, mix=args.mix,
              carry_root=carry_root)
    sets, feedback = _feedback(args, 'decision')
    given = {str(Path(c).resolve()) for c in args.cf}
    if feedback and not all(c and str(Path(c).resolve()) in given for c in feedback['cf']):
      parser.error('train-value takes --feedback built from its own --cf folders only')
    train = MultiQWindows(args.cf, 'train', cf_boost=args.cf_boost, seed=seed,
                          remedial=sets.get('remedial'), remedial_boost=args.remedial_boost,
                          correct=sets.get('correct'), correct_boost=args.correct_boost, **kw)
    held = MultiQWindows(args.cf, 'eval', **kw) if args.holdout else None
    constant = train.constant()
    report = {'cf': [str(c) for c in args.cf], 'policy': train.policy,
              'contracts': train.contracts, 'mix': args.mix, 'holdout': args.holdout,
              'diagnostic': diagnostic, 'window_start': 'carry' if carry_root else 'zero',
              'carry_root': carry_root, 'cf_boost': args.cf_boost, 'constant': constant,
              'train_pairs': train.pair_counts(), 'eval_pairs': held and held.pair_counts(),
              'eval': []}
    if feedback:
      report['feedback'] = feedback
    print('Q targets', {k: report[k] for k in ('constant', 'train_pairs', 'eval_pairs')})
    folder = _train(agent, args, train, held, report, constant)
    (folder / 'train_value.json').write_text(json.dumps(report, indent=1))
    print('Saved', folder)
  elif args.command == 'freeze':
    teaching = {}
    for name, data in reports(args.run).items():
      teaching[name] = {k: v for k, v in data.items() if k != 'eval'}
      teaching[name]['final_eval'] = data['eval'][-1] if data['eval'] else None
    if anchor:
      teaching['value_anchor'] = anchor
    lineage = {}
    if (Path(args.run) / 'inherit.json').exists():
      lineage['inherited_from'] = json.loads((Path(args.run) / 'inherit.json').read_text())['source']
    if args.cf:
      lineage.update(cf=[str(c) for c in args.cf], q_teacher_policy=sorted(
          {json.loads((Path(c) / 'meta.json').read_text())['policy'] for c in args.cf}))
    if args.value_anchor_of:
      lineage['value_kept_from'] = str(args.value_anchor_of)
    if args.feedback:
      from . import feedback as F
      F.summaries(args.feedback)
      lineage.update(feedback=[str(f) for f in args.feedback], teaching_order=F.OFFLINE_TRAIN,
                     time_ordered=F.TIME_ORDERED_NOT_SUPPORTED)
    print('Frozen', freeze(agent, args.run, args.name, args.preset, args.corpus or (),
                           teaching, lineage, args.out))
  else:
    out = Path(args.run) / 'rollouts'
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'rollout_{int(agent.n_actions):09d}.jsonl').open('w') as f:
      _, rows = rollout(agent, corpus_env(_one(args.corpus, '--corpus')), args.steps,
                        args.mode, record=f)
    adds = sum(r['executed'] == 1 for r in rows)
    print(f'rollout steps={len(rows)} adds={adds} '
          f'reward={sum(r["reward"] for r in rows):.3f}')
    if not latest(args.run):
      print('Saved', save(agent, args.run))


if __name__ == '__main__':
  main()
