"""Bounded live DAgger training with the original full-state lifecycle."""
import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
import uuid

import elements
import embodied
import jax
import numpy as np

from embodied.run.action_mask_warmup import METRICS
from embodied.run.fg_checkpoint import FullCheckpoint, RuntimeState, StepSchedule, file_hash, host, plain
from embodied.run.fg_s3_gate import make_gate_controller
from embodied.run.fg_s3_source_admission import transfer_control_state, runtime_versions
from embodied.run.fg_s3_source_transfer import make_agent, apply_and_audit_transfer
from embodied.run.fg_s3c_train import code_identity, reconcile_update_log
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json, check_files
from gym_trading_env.research.fg_s3_oracle_state_query import load_train_query
from gym_trading_env.research.fg_s3_v2_collector import V2Collector


PROTOCOL = 'FG_S3_V2_LIVE_LEARNING_V1'


def execution_code():
    code = code_identity()
    cli = Path(__file__).resolve().parents[3]/'gym-trading-env/tools/fg_s3_v2_experiment.py'
    code[str(cli)] = file_hash(cli)
    return code


class Counters:
    def __init__(self, mode='a2'):
        self.state = dict(all_updates=0, wm_updates=0, bc_valid_updates=0,
            bc_add_updates=0, rl_actor_updates=0, rl_critic_updates=0,
            gate_closed_updates=0, cycles=0, sampled_transitions=0)
        if mode == 'tutor':
            self.state.update(tutor_valid_update_count=0, pair_update_count=0,
                              pairable_batches=0, pair_count=0)

    def save(self):
        return copy.deepcopy(self.state)

    def load(self, state):
        self.state = copy.deepcopy(state)


def reconcile_cycles(root, count, env_steps, agent_updates):
    """Preserve an uncheckpointed cycle tail before restoring its boundary."""
    path = root/'cycles.jsonl'
    raw = path.read_bytes() if path.exists() else b''
    lines = raw.splitlines(keepends=True)
    if len(lines) < count:
        raise ValueError('Cycle log missing checkpointed cycles')
    rows = [json.loads(line) for line in lines[:count]]
    if any(a['env_steps'] >= b['env_steps'] or a['agent_updates'] >= b['agent_updates']
           for a, b in zip(rows, rows[1:])):
        raise ValueError('Checkpointed cycle log chronology differs')
    if rows and (rows[-1]['env_steps'] != env_steps or rows[-1]['agent_updates'] != agent_updates):
        raise ValueError('Cycle log differs from restored checkpoint boundary')
    committed = b''.join(lines[:count])
    if committed != raw:
        backup = root/'recovery_logs'; backup.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        (backup/('cycles_before_resume_'+token+'.jsonl')).write_bytes(raw)
        pending = root/('.cycles.pending_'+token)
        pending.write_bytes(committed)
        os.replace(pending, path)


class LiveRun:
    def __init__(self, prepared, run_dir, *, resume=False, authorize_training=False):
        self.prepared = Path(prepared).resolve()
        preparation = read_manifest(self.prepared/'manifest.json')
        check_files({str(self.prepared/n): h for n, h in preparation['files'].items()})
        if preparation['code'] != execution_code():
            raise ValueError('Prepared V2 code changed; create a new preparation')
        self.settings = s = read_manifest(self.prepared/'settings.json')['settings']
        from gym_trading_env.research.fg_s3_v2_experiment import require_training_authorization
        require_training_authorization(s, authorize_training)
        self.spec = spec = read_manifest(self.prepared/'source_admission.json')
        if s['device'] != 'gpu':
            raise ValueError('V2 model work is GPU-only')
        self.root = root = Path(run_dir).resolve()
        allowed = (Path('/data/logdir/fg_single3/s3_v2') if s['scope'] == 'formal' else
                   Path(__file__).resolve().parents[3]/'gym-trading-env/artifacts/fg_s3_v2/runs')
        if not root.is_relative_to(allowed):
            raise ValueError('V2 diagnostic output escaped its development run directory')
        if resume:
            saved = read_manifest(root/'run_identity.json')['identity']
            if (saved['preparation_sha256'] != file_hash(self.prepared/'manifest.json')
                    or saved['code'] != execution_code() or saved['packages'] != runtime_versions()):
                raise ValueError('Resume preparation/code/dependencies differ')
            check_files(saved['immutable_files'])
        else:
            root.mkdir(parents=True, exist_ok=False)
        if s['contracts'] is None:
            query, env, _, _, _ = load_train_query(s['env_receipt'], s['oracle_receipt'], s['contract'])
        else:
            from gym_trading_env.research.fg_s3_v2_train_pool import load_pool
            query, env = load_pool(s, self.prepared)
        if query.identity_hash != preparation['oracle_identity']:
            raise ValueError('Prepared Oracle identity changed')
        self.offset = int(spec.get('target_update_offset', spec['origin']['step']))
        frozen_model = read_manifest(self.prepared/'model_config.json')
        cfg = self.config = elements.Config(frozen_model['config'])
        extensions = agent_extensions(s)
        if s['mode'] == 'tutor' and frozen_model['agent_extension'] != extensions:
            raise ValueError('Frozen Tutor model configuration differs from settings')
        self.agent = agent = make_agent(plain(cfg), root/'model', device='gpu',
            agent_cls=S3V2Agent, bc_lambda=s['bc_lambda'], extra_agent_config=extensions)
        if not all(d.platform == 'gpu' for d in jax.devices()):
            raise RuntimeError('V2 model did not initialize on CUDA')
        if not resume:
            transfer = apply_and_audit_transfer(agent, spec)
            write_json(root/'source_transfer.json', transfer)
            control = transfer_control_state(spec)
        else:
            control = dict(gate=None, aggregate=None, samples=0)
        self.collector = collector = V2Collector(env, query, root/'rollout', s['dreamer_root'],
            seed=s['replay_seed'], beta=s['beta'], actor=lambda c, o: agent.policy(c, o, mode='train'),
            agent=agent, source_checkpoint_identity=spec['target_model_identity_hash'], existing=resume,
            replay_capacity=s['replay_capacity'], max_env_steps=s['env_steps'],
            beta_schedule=s['beta_schedule'])
        self.gate = make_gate_controller(plain(cfg), spec['gate_policy'], control,
                                        agent.get_avail_actor_enabled())
        source = embodied.streams.Consec(embodied.streams.Stateless(partial(collector.sample, cfg.batch_size)),
            length=cfg.batch_length, consec=1, prefix=cfg.replay_context, strict=True)
        self.stream = agent.stream(source)
        self.carry = dict(train=agent.init_train(cfg.batch_size), report=agent.init_report(cfg.batch_size))
        self.step = elements.Counter()
        self.counters = Counters(s['mode'])
        self.samples = [0 if resume else control['samples']]
        self.aggregates = {'gate': elements.Agg()}
        if not resume and control.get('aggregate'):
            self.aggregates['gate'].reducers = copy.deepcopy(control['aggregate'])
        self.schedule = StepSchedule(s['collect_every'], s['save_every'])
        def set_carry(data):
            self.carry = {k: agent.restore_carry(v) for k, v in data.items()}
        self.runtime = RuntimeState(self.schedule, elements.when.Ratio(1), [self.stream],
            lambda: self.carry, set_carry, self.aggregates, self.samples)
        if not resume:
            import yaml
            (root/'config.yaml').write_text(yaml.safe_dump(plain(cfg)))
            write_json(root/'source_admission.json', spec)
            identity = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, mode=s['mode'],
                budget_unit='environment_advances', env_steps=s['env_steps'],
                source_update_offset=self.offset, oracle_identity=query.identity_hash,
                preparation_sha256=file_hash(self.prepared/'manifest.json'),
                code=execution_code(), packages=runtime_versions(), device='gpu',
                source_kind=spec['kind'], source_model_identity=spec['target_model_identity_hash'],
                training_identity=preparation['training_identity'], scope=s['scope'],
                gate_policy=spec['gate_policy'], model_config_sha256=fingerprint(plain(cfg)),
                immutable_files={str(root/'source_transfer.json'): file_hash(root/'source_transfer.json'),
                                 str(root/'config.yaml'): file_hash(root/'config.yaml'),
                                 str(root/'source_admission.json'): file_hash(root/'source_admission.json')},
                seed=cfg.seed, mix_replay_seed=s['replay_seed'], beta=s['beta'],
                beta_schedule=s['beta_schedule'], bc_lambda=s['bc_lambda'],
                versions=preparation['versions'], diagnostic=s['scope'] == 'diagnostic')
            if s['mode'] == 'tutor':
                identity['agent_extension'] = extensions
            write_json(root/'run_identity.json', signed(dict(identity=identity)))
        else:
            identity = saved
        self.identity = identity
        self.cp = FullCheckpoint(root/'ckpt', self.step, identity, [self.stream],
                                 reserve_gib=10 if s['scope'] == 'formal' else 1, keep_all=True)
        self.cp.agent = agent
        self.cp.replay = collector.replay
        self.cp.driver = collector.driver
        self.cp.collector = collector
        self.cp.step = self.step
        self.cp.runtime = self.runtime
        self.cp.action_mask_warmup = self.gate
        self.cp.v2_counters = self.counters
        self.last_batch = None
        self.just_restored = resume
        if resume:
            self.cp.load()
            if (int(agent.n_updates) != self.offset + self.counters.state['all_updates']
                    or int(self.step) != collector.counts['env_steps']
                    or agent.get_avail_actor_enabled() != self.gate.actor_enabled):
                raise ValueError('Restored environment/update/Gate counters differ')
            reconcile_update_log(root, self.counters.state['all_updates'], self.offset)
            reconcile_cycles(root, self.counters.state['cycles'], int(self.step), int(agent.n_updates))
        else:
            self.cp.save()

    def update(self):
        if not len(self.collector.replay):
            raise ValueError('Collect at least 65 completed rows before sampling')
        self.runtime.start()
        batch = next(self.stream)
        ids = host(batch['stepid'])
        valid = host(batch['teacher_valid'])[:, 1:]
        y = host(batch['teacher_action'])[:, 1:]
        self.last_batch = dict(batch)  # Agent.train removes only its seed key.
        old_step = int(np.asarray(jax.device_get(self.agent.params['opt/step/value'])))
        self.carry['train'], _, mets = self.agent.train(self.carry['train'], batch)
        if not np.isfinite(float(mets['opt/loss'])) or not bool(mets.get('opt/finite', True)):
            raise FloatingPointError('V2 model update is nonfinite')
        new_step = int(np.asarray(jax.device_get(self.agent.params['opt/step/value'])))
        if new_step != old_step + 1:
            raise ValueError('V2 update must call the joint optimizer exactly once')
        if not bool(mets['v2/rl_active']) and bool(mets['opt/critic_changed']):
            raise RuntimeError('Inactive RL changed the Critic during a joint update')
        c = self.counters.state
        c['all_updates'] += 1
        c['wm_updates'] += 1
        c['bc_valid_updates'] += int(bool(mets['bc/active']))
        c['bc_add_updates'] += int(bool(mets['bc/active']) and (valid & (y == 1)).any())
        c['rl_actor_updates'] += int(bool(mets['v2/rl_active']))
        c['rl_critic_updates'] += int(bool(mets['v2/rl_active']))
        c['gate_closed_updates'] += int(not bool(mets['avail/ready']))
        c['sampled_transitions'] += int(valid.size)
        if self.settings['mode'] == 'tutor':
            c['tutor_valid_update_count'] += int(bool(mets['tutor/actor_active']))
            c['pair_update_count'] += int(bool(mets['tutor/pair_active']))
            c['pairable_batches'] += int(bool(mets['tutor/pairable_batch']))
            c['pair_count'] += int(mets['tutor/pair_count'])
        if int(self.agent.n_updates) != self.offset + c['all_updates']:
            raise ValueError('V2 source/local update counter drift')
        agg = self.aggregates['gate']
        # CLOSED training does not execute imagination; its placeholder zero
        # must never certify fallback for the Gate. Measure it in report below.
        for name, _ in METRICS.values():
            if name != 'avail/img_fallback_rate':
                agg.add(name, mets[name])
        for prefix in ('post', 'prior'):
            for suffix in ('add_positives', 'add_true_positives'):
                k = f'avail/{prefix}_{suffix}'
                agg.add(k, mets[k], agg='sum')
        self.samples[0] += self.config.batch_size * self.config.batch_length
        row = dict(local_update=c['all_updates'], agent_update=int(self.agent.n_updates),
            env_steps=int(self.step), valid_count=int(valid.sum()), add_count=int((valid & (y == 1)).sum()),
            stepid_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            sampled_stepids=ids.reshape(-1, 20).tolist(),
            teacher_sha256=hashlib.sha256(y.tobytes() + valid.tobytes()).hexdigest(),
            metrics={k: float(np.asarray(v)) for k, v in mets.items()
                     if np.asarray(v).size == 1 and np.isfinite(v).all()})
        positives = float(mets.get('bc/tp', 0)) + float(mets.get('bc/fn', 0))
        row['oracle_add_recall'] = float(mets['bc/tp'])/positives if positives else None
        if self.settings['mode'] == 'tutor':
            measured = bool(mets['tutor/scores_evaluated'])
            if not measured:
                row['oracle_add_recall'] = None
            # Missing groups are undefined, not a measured zero-quality score.
            for flag, keys in {
                    'pos': ('p_add_pos_mean', 'fn_rate'),
                    'neg': ('p_add_neg_mean', 'fp_rate'),
                    'pair': ('pair_accuracy',),
                }.items():
                if not measured or not bool(mets[f'tutor/{flag}_defined']):
                    for key in keys:
                        row['metrics'][f'tutor/{key}'] = None
            if not measured or not (bool(mets['tutor/pos_defined']) and bool(mets['tutor/neg_defined'])):
                row['metrics']['tutor/logit_gap_pos_neg'] = None
            row['tutor_pairable_batch_rate'] = c['pairable_batches']/c['all_updates']
            row['tutor_pair_update_count'] = c['pair_update_count']
            row['tutor_valid_update_count'] = c['tutor_valid_update_count']
        with (self.root/'updates.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        tutor_note = (f" pair={int(mets['tutor/pair_count'])} pair_loss={float(mets['tutor/pair_loss']):.6f}"
                      if self.settings['mode'] == 'tutor' else '')
        print(f"{self.settings['mode'].upper()} env={int(self.step)} update={c['all_updates']} BC={float(mets['bc/nll']):.6f} "
              f"valid={int(valid.sum())} ADD={int((valid & (y == 1)).sum())} "
              f"beta={self.collector.policy.beta:.6f} "
              f"actor_grad={float(mets.get('opt/actor_grad_norm', 0)):.6f}{tutor_note}", flush=True)
        return row

    def report_gate(self):
        if self.last_batch is None:
            return
        _, metrics = self.agent.report(self.carry['report'], dict(self.last_batch))
        self.aggregates['gate'].add('avail/img_fallback_rate', metrics['avail/img_fallback_rate'])
        if self.gate.ready(self.samples[0]):
            window = self.aggregates['gate'].result()
            self.samples[0] = 0
            for p in ('post', 'prior'):
                n = window.get(f'avail/{p}_add_positives', 0)
                window[f'avail/{p}_add_recall'] = window.get(f'avail/{p}_add_true_positives', 0)/n if n else float('nan')
            self.gate.update(window, int(self.agent.n_updates))
            self.agent.set_avail_actor_enabled(self.gate.actor_enabled)

    def run_to(self, target=None):
        s = self.settings
        target = s['env_steps'] if target is None else target
        if (type(target) is not int or target > s['env_steps'] or target < int(self.step)
                or (target != s['env_steps'] and target % s['collect_every'])):
            raise ValueError('Stop boundary must be a configured collection cycle within the frozen budget')
        started = time.perf_counter()
        while int(self.step) < target:
            cycle_started = time.perf_counter()
            end = min(int(self.step) + s['collect_every'], target)
            with self.stream.pause():
                self.collector.advance(end)
                self.step.increment(self.collector.counts['env_steps'] - int(self.step))
            if int(self.step) != end:
                raise RuntimeError('TRAIN contract ended before the configured diagnostic budget')
            collect_seconds = time.perf_counter() - cycle_started
            self.collector.prune()
            updates_started = time.perf_counter()
            audit = (self.just_restored or not self.counters.state['cycles']
                     or end % s['save_every'] == 0 or end == target)
            before = parameter_digests(self.agent.params) if audit else None
            cycle_rl_ready = self.gate.actor_enabled
            updated = []
            for _ in range(s['updates_per_collect']):
                updated.append(self.update())
            self.agent.sync_policy_from_train()
            self.report_gate()
            after = parameter_digests(self.agent.params) if audit else None
            changed = [k for k in before if before[k] != after[k]] if audit else None
            if audit and (s['mode'] != 'c2' or not cycle_rl_ready) and any(k.startswith(
                                       ('val/', 'slowval/', 'slowval_count/', 'retnorm/', 'valnorm/', 'advnorm/'))
                                       or (k.startswith('opt/state/') and '/val/' in k) for k in changed):
                raise RuntimeError('Non-RL update changed a frozen Critic/RL state')
            cycle = dict(env_steps=int(self.step), agent_updates=int(self.agent.n_updates), changed_keys=changed,
                         parameter_audited=audit,
                         actor_changed=any(r['metrics']['opt/actor_changed'] for r in updated),
                         wm_changed=any(r['metrics']['opt/wm_changed'] for r in updated),
                         counts=dict(self.collector.counts), gate=self.gate.metrics(),
                         beta_schedule=self.collector.beta_schedule.save(),
                         collect_seconds=collect_seconds, update_seconds=time.perf_counter()-updates_started,
                         retained_teacher_rows=len(self.collector.records), replay_items=len(self.collector.replay))
            with (self.root/'cycles.jsonl').open('a') as f:
                f.write(json.dumps(cycle)+'\n')
            self.counters.state['cycles'] += 1
            self.just_restored = False
            _, save = self.schedule.advance(self.step)
            if save:
                self.cp.save()
        self.cp.save(final=target == s['env_steps'])
        node = self.cp.latest()
        write_json(self.root/'env_state.json', self.collector.env.export_state())
        # Complete history is written once during publication. Checkpoints
        # contain the bounded teachers corresponding to all retained Replay.
        import resource
        elapsed = time.perf_counter()-started
        cycles = [json.loads(line) for line in (self.root/'cycles.jsonl').read_text().splitlines()]
        collect_seconds = sum(r['collect_seconds'] for r in cycles)
        update_seconds = sum(r['update_seconds'] for r in cycles)
        tp, fn = (self.collector.counts[k] for k in ('actor_oracle_ADD_tp', 'actor_oracle_ADD_fn'))
        result = signed(dict(protocol=PROTOCOL,
            status=(('FORMAL' if s['scope'] == 'formal' else 'DIAGNOSTIC') +
                    ('_COMPLETE' if target == s['env_steps'] else '_STOPPED')),
            run_dir=str(self.root), identity=self.identity, env_steps=int(self.step),
            counters=self.counters.save(), model_updates=int(self.agent.n_updates),
            counts=dict(self.collector.counts), reward=float(self.collector.counts['reward_sum']),
            beta_schedule=self.collector.beta_schedule.save(),
            checkpoint=str(node), checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
            devices=[str(d) for d in jax.devices()], elapsed_seconds=elapsed,
            collect_seconds=collect_seconds, update_seconds=update_seconds,
            core_env_steps_per_second=int(self.step)/max(collect_seconds+update_seconds, 1e-6),
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            oracle_query_recent_seconds={str(p): float(np.percentile(self.collector.query_seconds, p))
                for p in (50, 95, 99)},
            actor_oracle_add_recall=tp/(tp+fn) if tp+fn else None,
            retained_teacher_rows=len(self.collector.records), teacher_journal=str(self.collector.journal),
            actor_proposal_versions=sorted(self.collector.model_versions),
            label_ratio=self.collector.counts['certified_labels']/max(int(self.step), 1),
            formal_1m='RUN' if s['scope'] == 'formal' else 'NOT_RUN',
            test='NOT_RUN', independent_actor_success='NOT_ESTABLISHED'))
        if s['mode'] == 'tutor':
            c = self.counters.state
            result['tutor'] = dict(agent_extension=agent_extensions(s),
                pairable_batch_rate=c['pairable_batches']/max(c['all_updates'], 1),
                pair_update_count=c['pair_update_count'],
                tutor_valid_update_count=c['tutor_valid_update_count'],
                pair_count=c['pair_count'], critic_rl_frozen=True,
                classification_tp_meaning='Oracle label true positive; not a TP5 trade')
            result = signed({k: v for k, v in result.items() if k != 'content_hash'})
        write_json(self.root/'training_receipt.json', result)
        return result

    def close(self):
        # The original Prefetch worker is a bounded daemon without a close API.
        # All entrypoint invocations are separate processes; snapshot first.
        self.collector.close()
