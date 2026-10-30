"""Thinking Student V1 formal training (C4 4): a thin orchestrator over the
run.py and exam.py commands, driven by one config of configs/thinking_student/.

  python -m dreamerv3.thinking_student.formal_train pretrain --config FILE [--until L1|L2|L3]
  python -m dreamerv3.thinking_student.formal_train iterate  --config FILE [--rounds N] [--until cf]
  python -m dreamerv3.thinking_student.formal_train evaluate --config FILE --policy NAME [--train-smoke]
  python -m dreamerv3.thinking_student.formal_train run      --config FILE [--rounds N]
  python -m dreamerv3.thinking_student.formal_train status   --config FILE

pretrain  one student learns L1 -> L2 -> L3 from windows of every selected
          TRAIN contract (foundation/, inherited from paths.init_f1 or
          paths.init_from), is frozen as pi_0000, takes a short TRAIN smoke on
          two contracts and its DEV exam. --until stops after that stage.
iterate   rounds (B2-B8) on the newest frozen pi_k: per contract of the round
          its formal collection to the true terminal, check and CF
          (experience/pi_k/); in learners/pi_{k+1} L4 on the CF targets of
          pi_k only, then the S4 prior refresh (L5); pi_{k+1} frozen with the
          value anchor; its Teacher-free exam. --until cf stops the round
          after its CF (and state-check), before L4. From round
          iteration.l6.from_round on, L6 (S5) joins the round: the teaching
          items of pi_k (experience/pi_k/feedback/), an L2 review before L4,
          and in the review, L4 and L5 more windows around the correct
          answers of pi_k (correct_fraction, correct_boost; on by default)
          and, when configured, around its errors (remedial_fraction,
          remedial_boost; off by default, user 2026-10-28).
evaluate  the Teacher-free exam of a frozen policy: DEV on the VALIDATION rows
          dev.contracts of paths.markets, else DEV_NOT_CONFIGURED;
          --train-smoke adds the TRAIN smoke.
run       pretrain, then iterate.
status    progress read from the disk (writes nothing).

Every legal CF candidate gets up to iteration.cf_probe_steps (default 64).
iteration.cf_budget is the continuation quota for unfinished probes; the
sampled branches resume their Native/policy state under the same cf_max_steps
cap. Unselected probes are UNAVAILABLE_BUDGET without a Q label.
iteration.cf_min_coverage: when a contract's valid pairs cover a
smaller share of its forks, the round skips L4 and pi_{k+1} keeps the value
brain and anchor of pi_k (L4_SKIPPED_CF_NOT_REPRESENTATIVE). An exploration
config sets iteration.cf_coverage: exploratory; such a round then runs L4 on
its valid exact pairs as S3_Q_TARGETS_EXPLORATORY, with the contracts below
the threshold in L4_coverage. No label or coverage changes.

iteration.q_return: native (default) keeps the CF labels of the Native
Quality reward. net_pnl_h2760 (E2 R2, user 2026-10-30) writes the CF of
each contract to experience/pi_k/cf_native/ and relabels it into cf/ with
q_return.py (net trade PnL undiscounted within 2760 market minutes, with its
exactness checks); the S5 feedback, L4 and the freeze read cf/ as before.
The round record keeps the label statistics by coupling kind (q_return).

student.window_start 'carry' starts the L2-L5 windows from the causal
market carry of the continuous path (market_carry/<contract>/, computed
once before L2 or L3 from the market brain after L1, which no later stage
trains) instead of the zero state with a loss burn-in.

Every step runs an existing command in its own process (logs/). A rerun
continues from the disk: a stage trains its budget minus its recorded
progress, a finished collection or CF of the same frozen policy is kept, a
frozen policy is never rewritten, and the missing exams of the newest frozen
policy come before the round limit or a new round of a later call. Within
a call, the DEV exam of a new candidate runs alongside the next round's
collection and CF and is finished before the next exam starts and before
the call ends. run_state.json mirrors the progress for
reading and is never consulted to admit anything. Budgets and rounds come
from the config; nothing loops without a bound.

L6 (S5) teaches OFFLINE_TRAIN only: every valid feedback of the TRAIN
material, ready_at kept with each item. TIME_ORDERED is refused
(TIME_ORDERED_NOT_SUPPORTED): no round is a strict time-ordered online
learning result. L6 re-runs the L2, L4 and L5 updates; it never trains L1,
so the MarketCarry stores stay the deployed market memory.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from .corpus import corpus_sources, discover
from .feedback import OFFLINE_TRAIN, TIME_ORDERED_NOT_SUPPORTED, check_order, check_thresholds
from .q_return import NATIVE as Q_NATIVE, NET_PNL as Q_NET_PNL
from .run import latest, progress, reports

DEFAULTS = dict(
    # Relative paths are relative to the config file.
    paths=dict(markets=None, corpus_root=None, run_root=None, init_f1=None, init_from=None),
    corpora=dict(contracts='discover', mix='uniform_contract'),
    dev=dict(contracts=[]),
    # window_start 'carry': L2-L5 windows start from the run's MarketCarry
    # stores (market_carry/, carry_parallel processes compute them).
    student=dict(preset='size50m', device='cuda', seed=0, batch_size=8, batch_length=64,
                 log_every=100, window_start='zero', carry_parallel=1),
    foundation=dict(market_updates=0, outcome_updates=0, policy_updates=0, save_every=0,
                    holdout=0, demo_balance=1.0),
    iteration=dict(max_rounds=0, contracts=None, contracts_per_round=2, parallel_contracts=1,
                   cf_wait_forks=None, cf_workers=1, cf_max_steps=8192, cf_probe_steps=64, cf_budget=None,
                   cf_min_coverage=None, cf_coverage='representative', value_updates=0,
                   prior_refresh_updates=0, cf_boost=16.0, save_every=0, holdout=0,
                   beta_q=1.0, kappa_q='auto', kappa_floor=None, state_check_windows=0,
                   state_check_seed_offset=0, q_return=Q_NATIVE,
                   # L6 (S5) from round `from_round` on (None: off). The correct
                   # set is on by default, the error set off (fraction 0, boost 1).
                   l6=dict(from_round=None, order=OFFLINE_TRAIN, outcome_review_updates=0,
                           correct_fraction=0.25, correct_boost=4.0, remedial_fraction=0.,
                           remedial_boost=1.0, outcome_correct_prob=0.5,
                           decision_correct_margin=1.0, outcome_error_prob=0.1,
                           decision_error_margin=1.0)),
    evaluation=dict(random_seeds=5, smoke_contracts=2, smoke_steps=6000),
)
FOUNDATION = (('train-market', 'L1', 'market_updates'), ('train-outcome', 'L2', 'outcome_updates'),
              ('train-policy', 'L3', 'policy_updates'))
CHECKS = ('rule_violations', 'formal_violations', 'masked_adds', 'point_violations')


def load_config(path):
  """The YAML over DEFAULTS; an unknown section or key is an error."""
  path = Path(path).resolve()
  given = yaml.safe_load(path.read_text()) or {}
  config = copy.deepcopy(DEFAULTS)
  for section, values in given.items():
    if section not in config:
      raise ValueError(f'{path}: unknown section {section}')
    unknown = set(values or {}) - set(config[section])
    if unknown:
      raise ValueError(f'{path}: unknown keys {sorted(unknown)} in {section}')
    for key, value in (values or {}).items():
      if isinstance(config[section][key], dict):
        unknown = set(value or {}) - set(config[section][key])
        if unknown:
          raise ValueError(f'{path}: unknown keys {sorted(unknown)} in {section}.{key}')
        config[section][key].update(value or {})
      else:
        config[section][key] = value
  paths = config['paths']
  for key, value in paths.items():
    if value is not None:
      paths[key] = str((path.parent / value).resolve())
  for key in ('corpus_root', 'run_root'):
    if not paths[key]:
      raise ValueError(f'{path}: paths.{key} is required')
  if paths['init_f1'] and paths['init_from']:
    raise ValueError(f'{path}: give paths.init_f1 or paths.init_from, not both')
  if config['dev']['contracts'] and not paths['markets']:
    raise ValueError(f'{path}: dev.contracts needs paths.markets')
  if config['student']['window_start'] not in ('zero', 'carry'):
    raise ValueError(f'{path}: student.window_start is zero or carry')
  it = config['iteration']
  if it['cf_budget'] is not None and not (isinstance(it['cf_budget'], int) and it['cf_budget'] >= 0):
    raise ValueError(f'{path}: iteration.cf_budget is a number of forks >= 0 or null')
  if not isinstance(it['cf_probe_steps'], int) or not 1 <= it['cf_probe_steps'] <= 64:
    raise ValueError(f'{path}: iteration.cf_probe_steps is in [1, 64]')
  if it['cf_min_coverage'] is not None and not 0 < it['cf_min_coverage'] <= 1:
    raise ValueError(f'{path}: iteration.cf_min_coverage is in (0, 1] or null')
  if it['cf_coverage'] not in ('representative', 'exploratory'):
    raise ValueError(f'{path}: iteration.cf_coverage is representative or exploratory')
  if it['q_return'] not in (Q_NATIVE, Q_NET_PNL):
    raise ValueError(f'{path}: iteration.q_return is {Q_NATIVE} or {Q_NET_PNL}')
  l6 = it['l6']
  try:
    check_order(l6['order'])
  except NotImplementedError as e:
    raise ValueError(f'{path}: iteration.l6.order {l6["order"]}: {e}') from None
  if l6['from_round'] is not None and not (type(l6['from_round']) is int and l6['from_round'] >= 1):
    raise ValueError(f'{path}: iteration.l6.from_round is a round number from 1, or null')
  shares = [l6[f'{k}_fraction'] for k in ('correct', 'remedial')]
  if not all(0 <= x <= 1 for x in shares) or sum(shares) > 1:
    raise ValueError(f'{path}: iteration.l6 correct_fraction and remedial_fraction are in [0, 1] '
                     'and add up to at most 1')
  if not all(l6[f'{k}_boost'] > 0 for k in ('correct', 'remedial')):
    raise ValueError(f'{path}: iteration.l6 correct_boost and remedial_boost are positive')
  try:
    check_thresholds(*(l6[k] for k in ('outcome_error_prob', 'decision_error_margin',
                                       'outcome_correct_prob', 'decision_correct_margin')))
  except ValueError as e:
    raise ValueError(f'{path}: iteration.l6: {e}') from None
  return config


class Layout:
  """Where everything of one formal run lives."""

  def __init__(self, root):
    self.root = Path(root)
    self.foundation, self.policies = self.root / 'foundation', self.root / 'policies'
    self.state, self.logs = self.root / 'run_state.json', self.root / 'logs'
    self.carry = self.root / 'market_carry'

  def policy(self, k):
    return self.policies / f'pi_{k:04d}'

  def learner(self, k):
    return self.root / 'learners' / f'pi_{k:04d}'

  def experience(self, policy, kind, contract):
    return self.root / 'experience' / Path(policy).name / kind / contract

  def exam(self, policy, label, contract):
    return self.root / 'eval' / Path(policy).name / label / contract

  def log(self, name):
    self.logs.mkdir(parents=True, exist_ok=True)
    return self.logs / f'{name}.log'


def _call(module, argv, log):
  """One step in its own process, its output appended to `log`."""
  argv = [str(a) for a in argv]
  env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')
  with open(log, 'a') as f:
    f.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} {module} {" ".join(argv)}\n')
    f.flush()
    code = subprocess.call([sys.executable, '-u', '-m', f'dreamerv3.thinking_student.{module}', *argv],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
  if code:
    raise RuntimeError(f'{module} {argv[0]} failed with exit code {code}; see {log}')


def _call_many(jobs, parallel):
  """Independent steps, `parallel` at a time; every failure is reported."""
  with ThreadPoolExecutor(max(1, parallel)) as pool:
    futures = [pool.submit(_call, *job) for job in jobs]
  errors = [f.exception() for f in futures if f.exception()]
  if errors:
    raise RuntimeError('; '.join(map(str, errors)))


def _read(path):
  return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _write_state(lay, state):
  lay.root.mkdir(parents=True, exist_ok=True)
  tmp = lay.state.with_name(lay.state.name + '.tmp')
  tmp.write_text(json.dumps(state, indent=1))
  tmp.rename(lay.state)


def _state(lay, cfg):
  return _read(lay.state) or dict(
      run_id=lay.root.name, phase=None, step=None, round=0, contracts=[], frozen_policy=None,
      latest_candidate=None, best_dev=None, rounds={}, updates_by_stage={},
      seed=cfg['student']['seed'], latest_checkpoint=None)


def _step(lay, state, **update):
  state.update(update, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
  _write_state(lay, state)


def _updates(lay, r=None):
  """Updates of each stage: L1-L3 of the foundation; L4 and L5 (the prior
  refresh, on the L3 optimizer) of round r's learner."""
  f = progress(lay.foundation)
  out = dict(L1=f['L1'], L2=f['L2'], L3=f['L3'])
  if r:
    p = progress(lay.learner(r))
    out.update(L4=p['L4'], L5=p['L3'])
  return out


def _contracts(cfg):
  folders = discover(cfg['paths']['corpus_root'], cfg['corpora']['contracts'])
  return {s['contract']: f for s, f in zip(corpus_sources(folders), folders)}


def _student(cfg):
  s = cfg['student']
  return ['--preset', s['preset'], '--device', s['device'], '--seed', s['seed'],
          '--batch-size', s['batch_size'], '--batch-length', s['batch_length'],
          '--log-every', s['log_every'], '--mix', cfg['corpora']['mix']]


def _corpus_args(folders):
  return [a for f in folders for a in ('--corpus', f)]


def _windows(cfg, lay):
  """Where L2-L5 windows start: the zero state, or the run's MarketCarry."""
  return (['--window-start', 'carry', '--carry-root', lay.carry]
          if cfg['student']['window_start'] == 'carry' else [])


def _market_carry(cfg, lay, contracts, source):
  """The MarketCarry store of each contract that has none yet (meta.json is
  written last), computed by `source`: ['--run', foundation, *init] or
  ['--policy', pi_k]. L2-L5 never train the market brain, so the stores of
  the market brain after L1 serve every later stage and round."""
  if cfg['student']['window_start'] != 'carry':
    return
  s = cfg['student']
  _call_many([('run', ['market-carry', *source, '--corpus', folder, '--out', lay.carry,
                       '--preset', s['preset'], '--device', s['device']],
               lay.log(f'market_carry_{c}'))
              for c, folder in contracts.items() if not (lay.carry / c / 'meta.json').exists()],
             s['carry_parallel'])


def _frozen(policy):
  return (Path(policy) / 'policy.json').exists()


def _newest(lay):
  ks = [int(p.name[3:]) for p in lay.policies.glob('pi_[0-9][0-9][0-9][0-9]') if _frozen(p)]
  return max(ks) if ks else None


def _takes_smoke(cfg, k):
  """pi_0000 always takes the TRAIN smoke; a round's candidate only when no
  DEV is configured."""
  return k == 0 or not cfg['dev']['contracts']


def _exam_missing(cfg, lay, policy, train_smoke):
  """Whether a part of the exam of `policy`, as evaluate takes it, is not on
  the disk yet (exam.json is written last and whole)."""
  dev = cfg['dev']['contracts']
  outs = ([lay.exam(policy, 'DEV', c) / 'exam.json' for c in dev] if dev else
          [lay.root / 'eval' / Path(policy).name / 'DEV_NOT_CONFIGURED.json'])
  if train_smoke:
    outs += [lay.exam(policy, 'TRAIN_SMOKE', c) / 'exam.json'
             for c in list(_contracts(cfg))[:cfg['evaluation']['smoke_contracts']]]
  return not all(out.exists() for out in outs)


def _collected(out):
  meta = _read(Path(out) / 'meta.json')
  return bool(meta and meta['done'])


def _cf_done(out, policy):
  meta = _read(Path(out) / 'meta.json')
  if meta and Path(meta['policy']).resolve() != Path(policy).resolve():
    raise ValueError(f'{out} holds the CF of {meta["policy"]}, not of {policy}')
  return meta is not None


def _cf_record(meta):
  """One CF in the round record; coverage is the share of its forks with a
  valid ADD/WAIT pair (the strata stay in its meta.json)."""
  rec = {k: v for k, v in meta.items()
         if k in ('counts', 'valid_pairs', 'forks', 'delta_g', 'seconds', 'end_status')}
  if meta.get('budget'):
    rec['budget'] = {k: v for k, v in meta['budget'].items()
                     if k in ('forks', 'undecided', 'simulated', 'unavailable', 'reason')}
  chosen = meta['forks']['chosen']
  rec['coverage'] = round(meta['valid_pairs'] / chosen, 4) if chosen else None
  return rec


def _check_problems(out):
  report = _read(Path(out) / 'check.json')
  problems = [k for k in CHECKS if report[k]]
  if report.get('facts', {}).get('disagree'):
    problems.append('facts_disagree')
  if not _collected(out):
    problems.append('collection_not_done')
  return problems


def _teaching_items(cfg, lay, state, r, rec, policy, contracts, collect, cf):
  """L6 B1: the teaching items of pi_k on each usable contract of round r
  (regenerated from the current CF and pooled kappa); the --feedback arguments."""
  it, l6 = cfg['iteration'], cfg['iteration']['l6']
  out = {c: lay.experience(policy, 'feedback', c) for c in cf}
  kappa = [a for folder in cf.values() for a in ('--kappa-cf', folder)]
  floor = [] if it['kappa_floor'] is None else ['--kappa-floor', it['kappa_floor']]
  # A local CF supplement can change both this contract's pairs and the round's
  # pooled kappa. Existing summary.json is not evidence of current feedback.
  if out:
    _step(lay, state, step='feedback')
  for c in out:
    _call('feedback', ['--collection', collect[c], '--corpus', contracts[c], '--cf', cf[c], *kappa,
                       '--mix', cfg['corpora']['mix'], *floor, '--out', out[c],
                       '--order', l6['order'], '--outcome-error-prob', l6['outcome_error_prob'],
                       '--decision-error-margin', l6['decision_error_margin'],
                       '--outcome-correct-prob', l6['outcome_correct_prob'],
                       '--decision-correct-margin', l6['decision_correct_margin']],
          lay.log(f'r{r:03d}_feedback_{c}'))
  rec['feedback'] = {}
  for c, folder in out.items():
    summary = _read(folder / 'summary.json')
    rec['feedback'][c] = dict({k: summary[k] for k in ('sources', 'abstained', 'errors', 'correct')},
                              kappa_q=summary['thresholds']['kappa_q'])
  return [a for folder in out.values() for a in ('--feedback', folder)]


def _feedback_args(l6):
  """(train-outcome / train-policy, train-value) arguments of the feedback
  sets L6 uses: a set whose share is 0 (boost 1) passes nothing."""
  shares, boosts = [], []
  for kind in ('correct', 'remedial'):
    if l6[f'{kind}_fraction'] > 0:
      shares += [f'--{kind}-fraction', l6[f'{kind}_fraction']]
    if l6[f'{kind}_boost'] != 1:
      boosts += [f'--{kind}-boost', l6[f'{kind}_boost']]
  return shares, boosts


def _feedback_windows(learner):
  """What the windows of each L6 training command of the learner showed."""
  keys = ('windows', 'unique_label_rows', 'label_rows', 'unique_q_rows', 'unique_pairs',
          *(f'{kind}_{k}' for kind in ('correct', 'remedial')
            for k in ('windows', 'share', 'rows', 'rows_seen')))
  return {name: {k: v for k, v in data['curriculum'].items() if k in keys}
          for name, data in reports(learner).items() if data.get('feedback')}


def pretrain(cfg, lay, state, until=None):
  """until: stop after this foundation stage, before pi_0000 is frozen; a
  later pretrain goes on from there. A stage trains only before every stage
  and policy that stands on it: a budget raised afterwards is refused."""
  contracts, f, p = _contracts(cfg), cfg['foundation'], cfg['paths']
  corpora = _corpus_args(contracts.values())
  init = (['--init-f1', p['init_f1']] if p['init_f1'] else
          ['--init-from', p['init_from']] if p['init_from'] else [])
  pi0 = lay.policy(0)
  _step(lay, state, phase='foundation', contracts=list(contracts))
  for i, (command, stage, key) in enumerate(FOUNDATION):
    done = progress(lay.foundation)
    left = f[key] - done[stage]
    if left > 0:
      later = [s for _, s, _ in FOUNDATION[i + 1:] if done[s]] + [pi0.name] * _frozen(pi0)
      if stage == 'L1' and any(lay.carry.glob('*/meta.json')):
        later.append(lay.carry.name)
      if later:
        raise ValueError(f'foundation {stage} has {left} updates left, but {", ".join(later)} '
                         'already stands on it: a raised budget needs a new run')
      windows = [] if stage == 'L1' else _windows(cfg, lay)
      if windows:
        _step(lay, state, step='market_carry')
        _market_carry(cfg, lay, contracts, ['--run', lay.foundation, *init])
      _step(lay, state, step=stage)
      _call('run', [command, '--run', lay.foundation, *init, *corpora, '--updates', left,
                    '--save-every', f['save_every'], '--holdout', f['holdout'],
                    '--demo-balance', f['demo_balance'], *windows, *_student(cfg)],
            lay.log(f'foundation_{stage}'))
    if stage == until:
      ckpt = latest(lay.foundation)
      _step(lay, state, step=f'{stage}_done', updates_by_stage=_updates(lay),
            latest_checkpoint=ckpt and str(ckpt))
      print(f'pretrain stops after {stage} (--until); pi_0000 is not frozen yet')
      return
  if latest(lay.foundation) is None:
    raise RuntimeError(f'{lay.foundation} has no checkpoint: no foundation budget was trained')
  if not _frozen(pi0):
    _step(lay, state, step='freeze')
    _call('run', ['freeze', '--run', lay.foundation, '--name', pi0.name, '--out', pi0, *corpora,
                  *_student(cfg)], lay.log('freeze_pi_0000'))
  _step(lay, state, step='exam', frozen_policy=str(pi0),
        latest_candidate=state['latest_candidate'] or str(pi0), updates_by_stage=_updates(lay),
        latest_checkpoint=str(latest(lay.foundation)))
  evaluate(cfg, lay, state, pi0, train_smoke=_takes_smoke(cfg, 0))
  _step(lay, state, step='done')


def smoke(cfg, lay, policy, contracts):
  """TRAIN smoke of a frozen policy on the first contracts: the same
  formal student.policy on each, never an out-of-sample result."""
  e = cfg['evaluation']
  for name, folder in list(contracts.items())[:e['smoke_contracts']]:
    out = lay.exam(policy, 'TRAIN_SMOKE', name)
    if not (out / 'exam.json').exists():
      _call('exam', ['--policy', policy, '--out', out, '--corpus', folder,
                     '--steps', e['smoke_steps'], '--random-seeds', e['random_seeds']],
            lay.log(f'exam_{policy.name}_TRAIN_SMOKE_{name}'))


def _round_contracts(it, names, r):
  """The contracts of round r: the configured list, or the next
  `contracts_per_round` in rotation."""
  if it['contracts']:
    return list(it['contracts'])
  m = min(it['contracts_per_round'], len(names))
  return [names[((r - 1) * m + i) % len(names)] for i in range(m)]


class _Exams:
  """The DEV exams of a new candidate run in the background while the next
  round collects and runs its CF, which do not depend on them; join()
  finishes them before the next exam starts or iterate returns, and records
  best_dev in the calling thread (the only writer of run_state.json)."""

  def __init__(self, cfg, lay, state):
    self.cfg, self.lay, self.state = cfg, lay, state
    self.pool, self.pending = ThreadPoolExecutor(1), None

  def start(self, policy, train_smoke):
    self.join()
    cfg, lay = self.cfg, self.lay
    if train_smoke or not cfg['dev']['contracts']:
      evaluate(cfg, lay, self.state, policy, train_smoke)
      return
    e = cfg['evaluation']
    jobs = [('exam', ['--policy', policy, '--out', lay.exam(policy, 'DEV', c), '--pool',
                      cfg['paths']['markets'], '--contract', c, '--random-seeds', e['random_seeds']],
             lay.log(f'exam_{Path(policy).name}_DEV_{c}'))
            for c in cfg['dev']['contracts'] if not (lay.exam(policy, 'DEV', c) / 'exam.json').exists()]
    self.pending = (Path(policy), self.pool.submit(_call_many, jobs, len(jobs)))

  def running(self, policy):
    return self.pending is not None and self.pending[0] == Path(policy)

  def join(self):
    if self.pending is None:
      return
    (policy, future), self.pending = self.pending, None
    future.result()
    evaluate(self.cfg, self.lay, self.state, policy)  # the exams exist: best_dev only


def iterate(cfg, lay, state, rounds, until=None):
  """until='cf': the round stops after its CF and state-check, before L4; a
  later iterate goes on from there."""
  exams = _Exams(cfg, lay, state)
  try:
    _iterate(cfg, lay, state, rounds, until, exams)
  finally:
    exams.join()


def _iterate(cfg, lay, state, rounds, until, exams):
  contracts, it, s = _contracts(cfg), cfg['iteration'], cfg['student']
  corpora = _corpus_args(contracts.values())
  for _ in range(rounds):
    k = _newest(lay)
    if k is None:
      raise RuntimeError('No frozen policy: run pretrain first')
    policy = lay.policy(k)
    if not exams.running(policy) and _exam_missing(cfg, lay, policy, _takes_smoke(cfg, k)):
      # B7 of pi_k stopped after its freeze or part way: only its missing
      # exams, before the round limit or a new round. For k > 0 this
      # finishes round k, a round of this call.
      _step(lay, state, step='exam', frozen_policy=str(policy))
      evaluate(cfg, lay, state, policy, _takes_smoke(cfg, k))
      _step(lay, state, step='done')
      if k:
        continue
    if k >= it['max_rounds']:
      print(f'iteration.max_rounds = {it["max_rounds"]} reached at pi_{k:04d}')
      return
    r = k + 1
    rec = state['rounds'].setdefault(str(r), dict(
        policy_in=policy.name, contracts=_round_contracts(it, list(contracts), r)))
    _step(lay, state, phase='iteration', round=r, step='collect')
    # B2: the formal pi_k on each contract to the true terminal.
    collect = {c: lay.experience(policy, 'collect', c) for c in rec['contracts']}
    _call_many([('run', ['collect', '--policy', policy, '--corpus', contracts[c], '--out', out,
                         '--steps', 10 ** 9, '--chunk', 1024], lay.log(f'r{r:03d}_collect_{c}'))
                for c, out in collect.items() if not _collected(out)], it['parallel_contracts'])
    rec['collected'] = [c for c, out in collect.items() if _collected(out)]
    usable = []
    for c, out in collect.items():
      if not (out / 'check.json').exists():
        _call('run', ['check', '--out', out, '--corpus', contracts[c]], lay.log(f'r{r:03d}_check_{c}'))
      problems = _check_problems(out)
      rec.setdefault('check', {})[c] = problems or 'ok'
      if not problems:
        usable.append(c)
    if not usable:
      _step(lay, state, step='check_failed')
      raise RuntimeError(f'Round {r}: no collection of {policy.name} passed its check')
    # B3: same-policy ADD/WAIT counterfactuals of each usable contract.
    _step(lay, state, step='cf')
    cf = {c: lay.experience(policy, 'cf', c) for c in usable}
    relabel = it['q_return'] != Q_NATIVE
    native = {c: lay.experience(policy, 'cf_native', c) for c in usable} if relabel else cf
    forks = [] if it['cf_wait_forks'] is None else ['--forks', it['cf_wait_forks']]
    budget = [] if it['cf_budget'] is None else ['--budget', it['cf_budget']]
    _call_many([('run', ['cf', '--policy', policy, '--corpus', contracts[c], '--collection', collect[c],
                         '--out', out, '--workers', it['cf_workers'], '--max-steps', it['cf_max_steps'],
                         '--probe-steps', it['cf_probe_steps'],
                         '--seed', s['seed'], *forks, *budget], lay.log(f'r{r:03d}_cf_{c}'))
                for c, out in native.items() if not _cf_done(out, policy)], it['parallel_contracts'])
    if relabel:
      _step(lay, state, step='q_return')
      _call_many([('q_return', ['--collection', collect[c], '--cf', native[c], '--out', out],
                   lay.log(f'r{r:03d}_q_return_{c}'))
                  for c, out in cf.items() if not _cf_done(out, policy)], it['parallel_contracts'])
      rec['q_return'] = dict(name=it['q_return'], stats={
          c: _read(out / 'meta.json')['q_return']['stats'] for c, out in cf.items()})
    rec['cf'] = {c: _cf_record(_read(out / 'meta.json')) for c, out in cf.items()}
    # Missing stores only (pretrain made them; pi_k keeps the foundation's
    # market brain): the state-check, L4 and the L5 refresh read them.
    _market_carry(cfg, lay, contracts, ['--policy', policy])
    if it['state_check_windows']:
      check = collect[usable[0]] / 'state_check.json'
      if not check.exists():
        _step(lay, state, step='state_check')
        _call('run', ['state-check', '--policy', policy, '--collection', collect[usable[0]],
                      '--out', check, '--windows', it['state_check_windows'],
                      '--batch-length', s['batch_length'], '--seed', s['seed'],
                      '--seed-offset', it['state_check_seed_offset'], '--corpus',
                      contracts[usable[0]], *_windows(cfg, lay)],
              lay.log(f'r{r:03d}_state_check'))
    if until == 'cf':
      _step(lay, state, step='cf_done')
      print(f'iterate stops after the CF of round {r} (--until cf); L4 has not started')
      return
    # B4: L4 on the Q^{pi_k} targets only, unless a contract's valid pairs
    # cover too few of its forks to stand for its states: then the round
    # keeps pi_k's value brain and anchor; B5: the S4 prior refresh.
    learner, targets = lay.learner(r), [a for out in cf.values() for a in ('--cf', out)]
    common = ['--run', learner, '--init-from', policy, '--save-every', it['save_every'],
              '--holdout', it['holdout'], *_windows(cfg, lay), *_student(cfg)]
    floor = it['cf_min_coverage']
    low = {c: m['coverage'] for c, m in rec['cf'].items()
           if floor is not None and m['coverage'] is not None and m['coverage'] < floor}
    # An exploration config (iteration.cf_coverage: exploratory) still
    # teaches the valid exact pairs of such a round and records it so.
    thin = {} if it['cf_coverage'] == 'exploratory' else low
    rec['L4'] = 'L4_SKIPPED_CF_NOT_REPRESENTATIVE' if thin else (
        ('S3_Q_TARGETS_EXPLORATORY' if low else 'S3_Q_TARGETS') if it['value_updates']
        else 'L4_SKIPPED_BY_CONFIG')
    if low:
      rec['L4_coverage'] = dict(min_coverage=floor, below=low, cf_coverage=it['cf_coverage'])
    # L6 (S5): teaching items, the L2 review and the correct (and error)
    # windows in L4 and L5; without representative pairs the round takes no
    # decision feedback either (it comes from the same CF).
    l6, feedback, review, boost, demo = it['l6'], [], [], [], []
    if l6['from_round'] is not None and r >= l6['from_round']:
      feedback = _teaching_items(cfg, lay, state, r, rec, policy, contracts, collect, cf)
      review, boost = _feedback_args(l6)
      demo = [] if thin else [*feedback, *review]
      done = progress(learner)
      left = l6['outcome_review_updates'] - done['L2']
      if left > 0:
        if done['L4'] or done['L3']:
          raise ValueError(f'Round {r}: the L2 review has {left} updates left, but L4 or L5 of '
                           f'{learner.name} already stands on it: a raised budget needs a new run')
        _step(lay, state, step='L6_L2_review')
        _call('run', ['train-outcome', *common, *corpora, '--updates', left, *feedback, *review],
              lay.log(f'r{r:03d}_L6_L2'))
      rec['L6'] = 'S5_FEEDBACK_CURRICULUM'
      rec['l6'] = dict(order=l6['order'], time_ordered=TIME_ORDERED_NOT_SUPPORTED,
                       from_round=l6['from_round'], feedback=[str(f) for f in feedback[1::2]],
                       **{k: l6[k] for k in ('correct_fraction', 'correct_boost',
                                             'remedial_fraction', 'remedial_boost',
                                             'outcome_correct_prob', 'decision_correct_margin',
                                             'outcome_error_prob', 'decision_error_margin')},
                       decision_feedback=(
                           'SKIPPED_CF_NOT_REPRESENTATIVE' if thin else 'L4_L5'))
    else:
      rec['L6'] = 'L6_SKIPPED_BY_CONFIG'
    left = 0 if thin else it['value_updates'] - progress(learner)['L4']
    if left > 0:
      _step(lay, state, step='L4')
      _call('run', ['train-value', *common, *targets, '--updates', left, '--cf-boost', it['cf_boost'],
                    *feedback, *boost], lay.log(f'r{r:03d}_L4'))
    rec['L5'] = 'S4_PRIOR_REFRESH' if it['prior_refresh_updates'] else 'L5_SKIPPED_BY_CONFIG'
    left = it['prior_refresh_updates'] - progress(learner)['L3']
    if left > 0:
      _step(lay, state, step='L5')
      _call('run', ['train-policy', *common, *corpora, '--updates', left, *demo],
            lay.log(f'r{r:03d}_L5'))
    if feedback:
      rec['l6'].update(L2_review_updates=progress(learner)['L2'],
                       feedback_windows=_feedback_windows(learner))
    if latest(learner) is None:
      raise RuntimeError(f'Round {r}: {learner} trained nothing (value and refresh budgets are 0)')
    # B6: pi_{k+1} with the value anchor; pi_k stays as it was.
    new = lay.policy(r)
    if not _frozen(new):
      _step(lay, state, step='freeze')
      kappa = [] if it['kappa_floor'] is None else ['--kappa-floor', it['kappa_floor']]
      anchor = (['--value-anchor-of', policy] if thin else
                [*targets, '--beta-q', it['beta_q'], '--kappa-q', it['kappa_q'], *kappa])
      _call('run', ['freeze', '--run', learner, '--name', new.name, '--out', new, *corpora, *anchor,
                    '--holdout', it['holdout'], *feedback, *_student(cfg)],
            lay.log(f'r{r:03d}_freeze'))
    p, anchor = _updates(lay, r), _read(new / 'policy.json')['teaching'].get('value_anchor', {})
    rec.update(policy_out=new.name, updates=dict(L4=p['L4'], L5=p['L5']), value_anchor=dict(
        beta_q=anchor.get('beta_q'), kappa_q=anchor.get('kappa_q'),
        pairs=anchor.get('pairs', {}).get('total'), status=anchor.get('status')))
    _step(lay, state, step='exam', frozen_policy=str(new), latest_candidate=str(new),
          updates_by_stage=p, latest_checkpoint=str(latest(learner)))
    # B7: the Teacher-free exam of the new candidate, alongside the next round.
    exams.start(new, _takes_smoke(cfg, r))
    _step(lay, state, step='exam_running' if exams.pending else 'done')


def evaluate(cfg, lay, state, policy, train_smoke=False):
  policy, d, e = Path(policy), cfg['dev'], cfg['evaluation']
  if not _frozen(policy):
    raise ValueError(f'{policy} is not a frozen policy')
  if d['contracts']:
    for c in d['contracts']:
      out = lay.exam(policy, 'DEV', c)
      if not (out / 'exam.json').exists():
        _call('exam', ['--policy', policy, '--out', out, '--pool', cfg['paths']['markets'],
                       '--contract', c, '--random-seeds', e['random_seeds']],
              lay.log(f'exam_{policy.name}_DEV_{c}'))
    _best_dev(cfg, lay, state)
  else:
    out = lay.root / 'eval' / policy.name / 'DEV_NOT_CONFIGURED.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(label='DEV_NOT_CONFIGURED', policy=str(policy), reason=(
        'dev.contracts is empty; a TRAIN smoke is no DEV result')), indent=1))
    print('DEV_NOT_CONFIGURED: no DEV exam of', policy.name)
  if train_smoke:
    smoke(cfg, lay, policy, _contracts(cfg))


def _best_dev(cfg, lay, state):
  """best_dev: the frozen policy with the highest DEV reward per market
  minute (the preset main metric) over complete exams of every DEV contract;
  latest_candidate is kept apart."""
  scores = {}
  for policy in sorted(lay.policies.glob('pi_*')):
    reports = [_read(lay.exam(policy, 'DEV', c) / 'exam.json') for c in cfg['dev']['contracts']]
    if reports and all(r and r['complete'] for r in reports):
      scores[policy.name] = (sum(r['student']['reward'] for r in reports)
                             / sum(r['student']['minutes'] for r in reports))
  if scores:
    best = max(scores, key=scores.get)
    state['best_dev'] = dict(policy=str(lay.policies / best), value=scores[best], scores=scores,
                             metric='DEV reward per market minute')
    _write_state(lay, state)


def status(cfg, lay):
  """What the disk holds (read-only)."""
  out = dict(run_root=str(lay.root), state=_read(lay.state),
             foundation=dict(progress=progress(lay.foundation),
                             latest=str(latest(lay.foundation) or '')),
             policies={p.name: _frozen(p) for p in sorted(lay.policies.glob('pi_*'))},
             market_carry=sorted(m.parent.name for m in lay.carry.glob('*/meta.json')),
             experience={}, learners={}, exams={})
  for folder in sorted((lay.root / 'experience').glob('pi_*')):
    out['experience'][folder.name] = dict(
        collect={c.name: _collected(c) for c in sorted((folder / 'collect').glob('*'))},
        cf={c.name: (c / 'meta.json').exists() for c in sorted((folder / 'cf').glob('*'))},
        feedback={c.name: (c / 'summary.json').exists()
                  for c in sorted((folder / 'feedback').glob('*'))})
  for folder in sorted((lay.root / 'learners').glob('pi_*')):
    out['learners'][folder.name] = progress(folder)
  for exam in sorted((lay.root / 'eval').glob('*/*/*/exam.json')):
    r = _read(exam)
    out['exams']['/'.join(exam.parts[-4:-1])] = dict(
        complete=r['complete'], minutes=r['student']['minutes'], reward=r['student']['reward'],
        random_reward_mean=r['random']['summary'].get('reward', {}).get('mean'))
  return out


def _record_config(lay, cfg):
  """run_root/config.yaml: the resolved config of the latest call; an
  earlier different one is kept beside it with its date."""
  lay.root.mkdir(parents=True, exist_ok=True)
  file, text = lay.root / 'config.yaml', yaml.safe_dump(cfg, sort_keys=False)
  if file.exists() and file.read_text() != text:
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(file.stat().st_mtime))
    file.rename(lay.root / f'config.{stamp}.yaml')
  file.write_text(text)


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('command', choices=['pretrain', 'iterate', 'evaluate', 'run', 'status'])
  parser.add_argument('--config', required=True)
  parser.add_argument('--rounds', type=int, default=1, help='iterate/run: rounds of this call')
  parser.add_argument('--policy', default=None, help='evaluate: frozen policy name or folder')
  parser.add_argument('--train-smoke', action='store_true', help='evaluate: also the TRAIN smoke')
  stops = {'pretrain': [stage for _, stage, _ in FOUNDATION], 'iterate': ['cf']}
  parser.add_argument('--until', choices=[u for us in stops.values() for u in us], default=None,
                      help='pretrain: stop after this foundation stage, before freezing pi_0000; '
                           'iterate: cf stops the round after its CF, before L4')
  args = parser.parse_args(argv)
  if args.until and args.until not in stops.get(args.command, []):
    parser.error(f'--until {args.until} does not belong to {args.command}')
  cfg = load_config(args.config)
  lay = Layout(cfg['paths']['run_root'])
  if args.command == 'status':
    print(json.dumps(status(cfg, lay), indent=1))
    return
  if args.command == 'evaluate' and not args.policy:
    parser.error('evaluate needs --policy')
  _record_config(lay, cfg)
  state = _state(lay, cfg)
  state['config'] = str(Path(args.config).resolve())
  if args.command in ('pretrain', 'run'):
    pretrain(cfg, lay, state, args.until)
  if args.command in ('iterate', 'run'):
    iterate(cfg, lay, state, args.rounds, args.until if args.command == 'iterate' else None)
  if args.command == 'evaluate':
    policy = Path(args.policy) if '/' in args.policy else lay.policies / args.policy
    evaluate(cfg, lay, state, policy, args.train_smoke)


if __name__ == '__main__':
  main()
