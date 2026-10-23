"""Thinking Student V1 formal training (C4 4): a thin orchestrator over the
run.py and exam.py commands, driven by one config of configs/thinking_student/.

  python -m dreamerv3.thinking_student.formal_train pretrain --config FILE [--until L1|L2|L3]
  python -m dreamerv3.thinking_student.formal_train iterate  --config FILE [--rounds N]
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
          value anchor; its Teacher-free exam.
evaluate  the Teacher-free exam of a frozen policy: DEV on the VALIDATION rows
          dev.contracts of paths.markets, else DEV_NOT_CONFIGURED;
          --train-smoke adds the TRAIN smoke.
run       pretrain, then iterate.
status    progress read from the disk (writes nothing).

Every step runs an existing command in its own process (logs/). A rerun
continues from the disk: a stage trains its budget minus its recorded
progress, a finished collection or CF of the same frozen policy is kept, a
frozen policy is never rewritten, and the missing exams of the newest frozen
policy come before the round limit or a new round (a round ends with its
candidate's exam). run_state.json mirrors the progress for
reading and is never consulted to admit anything. Budgets and rounds come
from the config; nothing loops without a bound. S5 (L6: ready_at feedback
and the curriculum) is not implemented, and every round records that.
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
from .run import latest, progress

DEFAULTS = dict(
    # Relative paths are relative to the config file.
    paths=dict(markets=None, corpus_root=None, run_root=None, init_f1=None, init_from=None),
    corpora=dict(contracts='discover', mix='uniform_contract'),
    dev=dict(contracts=[]),
    student=dict(preset='size50m', device='cuda', seed=0, batch_size=8, batch_length=64,
                 log_every=100),
    foundation=dict(market_updates=0, outcome_updates=0, policy_updates=0, save_every=0,
                    holdout=0, demo_balance=1.0),
    iteration=dict(max_rounds=0, contracts=None, contracts_per_round=2, parallel_contracts=1,
                   cf_wait_forks=None, cf_workers=1, cf_max_steps=8192, value_updates=0,
                   prior_refresh_updates=0, cf_boost=16.0, save_every=0, holdout=0,
                   beta_q=1.0, kappa_q='auto', kappa_floor=None, state_check_windows=0),
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
    config[section].update(values or {})
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
  return config


class Layout:
  """Where everything of one formal run lives."""

  def __init__(self, root):
    self.root = Path(root)
    self.foundation, self.policies = self.root / 'foundation', self.root / 'policies'
    self.state, self.logs = self.root / 'run_state.json', self.root / 'logs'

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


def _check_problems(out):
  report = _read(Path(out) / 'check.json')
  problems = [k for k in CHECKS if report[k]]
  if report.get('facts', {}).get('disagree'):
    problems.append('facts_disagree')
  if not _collected(out):
    problems.append('collection_not_done')
  return problems


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
      if later:
        raise ValueError(f'foundation {stage} has {left} updates left, but {", ".join(later)} '
                         'already stands on it: a raised budget needs a new run')
      _step(lay, state, step=stage)
      _call('run', [command, '--run', lay.foundation, *init, *corpora, '--updates', left,
                    '--save-every', f['save_every'], '--holdout', f['holdout'],
                    '--demo-balance', f['demo_balance'], *_student(cfg)],
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


def iterate(cfg, lay, state, rounds):
  contracts, it, s = _contracts(cfg), cfg['iteration'], cfg['student']
  corpora = _corpus_args(contracts.values())
  for _ in range(rounds):
    k = _newest(lay)
    if k is None:
      raise RuntimeError('No frozen policy: run pretrain first')
    policy = lay.policy(k)
    if _exam_missing(cfg, lay, policy, _takes_smoke(cfg, k)):
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
    forks = [] if it['cf_wait_forks'] is None else ['--forks', it['cf_wait_forks']]
    _call_many([('run', ['cf', '--policy', policy, '--corpus', contracts[c], '--collection', collect[c],
                         '--out', out, '--workers', it['cf_workers'], '--max-steps', it['cf_max_steps'],
                         '--seed', s['seed'], *forks], lay.log(f'r{r:03d}_cf_{c}'))
                for c, out in cf.items() if not _cf_done(out, policy)], it['parallel_contracts'])
    rec['cf'] = {c: {k: v for k, v in _read(out / 'meta.json').items()
                     if k in ('counts', 'valid_pairs', 'forks', 'delta_g', 'seconds')}
                 for c, out in cf.items()}
    if it['state_check_windows']:
      check = collect[usable[0]] / 'state_check.json'
      if not check.exists():
        _step(lay, state, step='state_check')
        _call('run', ['state-check', '--policy', policy, '--collection', collect[usable[0]],
                      '--out', check, '--windows', it['state_check_windows'],
                      '--batch-length', s['batch_length'], '--seed', s['seed']],
              lay.log(f'r{r:03d}_state_check'))
    # B4: L4 on the Q^{pi_k} targets only; B5: the S4 prior refresh.
    learner, targets = lay.learner(r), [a for out in cf.values() for a in ('--cf', out)]
    common = ['--run', learner, '--init-from', policy, '--save-every', it['save_every'],
              '--holdout', it['holdout'], *_student(cfg)]
    left = it['value_updates'] - progress(learner)['L4']
    if left > 0:
      _step(lay, state, step='L4')
      _call('run', ['train-value', *common, *targets, '--updates', left, '--cf-boost', it['cf_boost']],
            lay.log(f'r{r:03d}_L4'))
    rec['L5'] = 'S4_PRIOR_REFRESH' if it['prior_refresh_updates'] else 'L5_SKIPPED_BY_CONFIG'
    left = it['prior_refresh_updates'] - progress(learner)['L3']
    if left > 0:
      _step(lay, state, step='L5')
      _call('run', ['train-policy', *common, *corpora, '--updates', left], lay.log(f'r{r:03d}_L5'))
    rec['L6'] = 'L6_NOT_IMPLEMENTED'
    if latest(learner) is None:
      raise RuntimeError(f'Round {r}: {learner} trained nothing (value and refresh budgets are 0)')
    # B6: pi_{k+1} with the value anchor; pi_k stays as it was.
    new = lay.policy(r)
    if not _frozen(new):
      _step(lay, state, step='freeze')
      floor = [] if it['kappa_floor'] is None else ['--kappa-floor', it['kappa_floor']]
      _call('run', ['freeze', '--run', learner, '--name', new.name, '--out', new, *targets, *corpora,
                    '--beta-q', it['beta_q'], '--kappa-q', it['kappa_q'], *floor,
                    '--holdout', it['holdout'], *_student(cfg)], lay.log(f'r{r:03d}_freeze'))
    p, anchor = _updates(lay, r), _read(new / 'policy.json')['teaching'].get('value_anchor', {})
    rec.update(policy_out=new.name, updates=dict(L4=p['L4'], L5=p['L5']), value_anchor=dict(
        beta_q=anchor.get('beta_q'), kappa_q=anchor.get('kappa_q'),
        pairs=anchor.get('pairs', {}).get('total'), status=anchor.get('status')))
    _step(lay, state, step='exam', frozen_policy=str(new), latest_candidate=str(new),
          updates_by_stage=p, latest_checkpoint=str(latest(learner)))
    # B7: the Teacher-free exam of the new candidate.
    evaluate(cfg, lay, state, new, train_smoke=_takes_smoke(cfg, r))
    _step(lay, state, step='done')


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
             experience={}, learners={}, exams={})
  for folder in sorted((lay.root / 'experience').glob('pi_*')):
    out['experience'][folder.name] = dict(
        collect={c.name: _collected(c) for c in sorted((folder / 'collect').glob('*'))},
        cf={c.name: (c / 'meta.json').exists() for c in sorted((folder / 'cf').glob('*'))})
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
  parser.add_argument('--until', choices=[stage for _, stage, _ in FOUNDATION], default=None,
                      help='pretrain: stop after this foundation stage, before freezing pi_0000')
  args = parser.parse_args(argv)
  if args.until and args.command != 'pretrain':
    parser.error('--until belongs to pretrain')
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
    iterate(cfg, lay, state, args.rounds)
  if args.command == 'evaluate':
    policy = Path(args.policy) if '/' in args.policy else lay.policies / args.policy
    evaluate(cfg, lay, state, policy, args.train_smoke)


if __name__ == '__main__':
  main()
