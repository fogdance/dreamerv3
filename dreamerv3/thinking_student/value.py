"""S3: policy-conditioned Q teaching material of a frozen complete student pi_k
(C3 5.5, A4, A6 2.4).

reference  pi_k's formal TRAIN collection (run.collect): one decision per
           market cursor from the first minute to the true terminal, as a gym
           ts_v1_cf.Reference with its complete discounted Native returns.
cf         same-state ADD/WAIT branches from legal flat decisions of the
           reference, continued by the same frozen pi_k on its recorded device
           and dtype; coupled branches get exact returns (ts_v1_cf).
targets    per reference row: q_real, the executed action's complete Native
           return (STUDENT_EXECUTED_NATIVE); per fork row: q_cf, the other
           action's branch return (VERIFIED_CF_NATIVE, VALID_COMPLETE only).
QWindows   L4 training windows over the reference rows.

Every CF worker first replays the whole reference in its own process and
refuses any action or reward that differs from the collection: the branches
and the coupling witness then compare against decisions of the same
compiled policy. The targets belong to pi_k only (A6 2.4): a new policy needs
its own collection and CF.

  python -m dreamerv3.thinking_student.run cf --policy DIR --corpus DIR [--forks N]
"""
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gym_trading_env.research import ts_v1_cf as CF

from . import labels as L

ADD, WAIT = CF.ADD, CF.WAIT
Q_KEYS = ('q_real', 'q_real_valid', 'q_cf', 'q_cf_valid')
WINDOW_OBS = ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m',
              'position_state', 'action_mask', 'account_native')


def load_collection(folder):
  """Rows and decision arrays of a formal pi_k collection; refuses forced,
  explored or behavior-policy rows (they are not Q^{pi_k} samples)."""
  folder = Path(folder)
  rows = [json.loads(line) for file in sorted((folder / 'steps').glob('*.jsonl'))
          for line in file.open()]
  parts = [dict(np.load(f)) for f in sorted((folder / 'replay').glob('*.npz'))]
  arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
  if len(rows) != len(arrays['action']):
    raise ValueError(f'{folder}: {len(rows)} step rows for {len(arrays["action"])} replay rows')
  if any(r['mode'] != 'policy' or r['forced'] or r['explored'] for r in rows):
    raise ValueError(f'{folder}: Q^pi_k needs the formal pi_k trajectory')
  with (folder / 'state.pkl').open('rb') as f:
    run = pickle.load(f)['run']
  return rows, arrays, run


def build_reference(folder, env):
  """The collection as a ts_v1_cf.Reference (complete only when the
  collection ended at the true terminal)."""
  rows, arrays, run = load_collection(folder)
  terminal = bool(run['done'] and run['env']['terminated'])
  return CF.Reference(
      env.market_clock_counter, [r['cursor'] for r in rows], arrays['reward'],
      [r['equity'] for r in rows], {k: arrays[k] for k in CF.INPUTS},
      arrays['action'], terminal, CF.coupling_floor(env)), rows


def choose_forks(reference, forks=None, seed=0):
  """Legal flat decisions of the reference: every pi_k ADD and all WAITs, or
  `forks` WAITs sampled uniformly (never by outcome, C3 5.5)."""
  cursors = reference.forks()
  own = reference.action[cursors - reference.start]
  adds, waits = cursors[own == ADD], cursors[own == WAIT]
  if forks is not None and forks < len(waits):
    waits = np.sort(np.random.default_rng(seed).choice(waits, forks, replace=False))
  return sorted(int(t) for t in np.concatenate([adds, waits]))


class Stepper:
  """ts_v1_cf stepper over run.Autonomous (the frozen pi_k on a Native env)."""

  def __init__(self, run):
    self.run, self.first_trade = run, len(run.env.attributions)

  def state(self):
    env = self.run.env
    return dict(cursor=int(env.current_step), equity=float(env._calculate_equity()),
                **{k: self.run.obs[k] for k in CF.INPUTS})

  def step(self, action=None):
    row, arrays = self.run.step(action)
    return row['reward'], self.run.done, bool(arrays['terminal'])

  def peek(self):
    """The decision the policy would take now, without stepping or using a
    policy seed."""
    from .run import student_obs
    run, agent = self.run, self.run.agent
    counter = int(agent.n_actions)
    obs = student_obs(run.obs, run.reward, run.first, False, False)
    _, act, out = agent.policy(run.carry, obs, mode=run.mode)
    agent.n_actions.value = counter
    return float(out['answer/z'][0]), int(act['action'][0])

  def trades(self):
    keys = ('entry_cursor', 'exit_cursor', 'reason', 'entry_price', 'exit_price',
            'net_trade_pnl')
    return [{k: t.get(k) for k in keys} for t in self.run.env.attributions[self.first_trade:]]


def _cf_job(job):
  """One CF worker: replay the reference, then branch at its forks."""
  os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  from .run import Autonomous, corpus_env, load_policy
  began = time.time()
  agent, meta = load_policy(job['policy'])
  env, branch_env = corpus_env(job['corpus']), corpus_env(job['corpus'])
  reference, recorded = build_reference(job['collection'], env)
  forks, saved, own = set(job['forks']), {}, {}
  main = Autonomous(agent, env, 'policy')
  z_differs = 0
  while not main.done:
    t = int(env.current_step)
    if t in forks:
      saved[t] = main.state()
    row, _ = main.step()
    rec = recorded[t - reference.start]
    if (row['cursor'], row['executed'], row['reward']) != (rec['cursor'], rec['executed'], rec['reward']):
      raise RuntimeError(f'Worker {job["index"]}: the replay differs from the collection at cursor {t}')
    z_differs += row['z'] != rec['z']
    own[t] = (row['z'], row['executed'])
  replay_seconds = time.time() - began
  out = []
  for t in sorted(forks):
    run = Autonomous.restore(agent, branch_env, saved.pop(t))
    stepper = Stepper(run)
    first = 1 - int(reference.action[t - reference.start])
    result = CF.run_branch(stepper, reference, first, job['max_steps'],
                           witness=lambda c: stepper.peek() == own[c])
    result.update(CF.pair(reference, result), trades=stepper.trades())
    out.append(result)
    if job['log_every'] and len(out) % job['log_every'] == 0:
      print(f'worker {job["index"]}: {len(out)}/{len(forks)} branches', flush=True)
  return dict(index=job['index'], results=out, replay_steps=len(own), z_differs=int(z_differs),
              replay_seconds=round(replay_seconds, 1), seconds=round(time.time() - began, 1),
              platform=agent.config.jax.platform, compute_dtype=agent.config.jax.compute_dtype)


def run_cf(policy, corpus, collection, out, forks=None, seed=0, workers=4,
           max_steps=8192, log_every=50, log=print):
  """Write out/pairs.jsonl, out/targets.npz and out/meta.json."""
  from .run import corpus_env
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  began = time.time()
  reference, _ = build_reference(collection, corpus_env(corpus))
  if not reference.complete:
    raise ValueError('The collection did not reach the true terminal: no complete returns')
  chosen = choose_forks(reference, forks, seed)
  groups = [chosen[i::workers] for i in range(workers)]
  jobs = [dict(index=i, policy=str(policy), corpus=str(corpus), collection=str(collection),
               forks=g, max_steps=max_steps, log_every=log_every)
          for i, g in enumerate(groups) if g]
  log(f'CF: {len(chosen)} forks of {len(reference.forks())} legal flat decisions, '
      f'{len(jobs)} workers')
  with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
    parts = list(pool.map(_cf_job, jobs))
  results = sorted((r for p in parts for r in p['results']), key=lambda r: r['cursor'])
  with (out / 'pairs.jsonl').open('w') as f:
    f.writelines(json.dumps(r) + '\n' for r in results)
  targets = cf_targets(reference, results)
  np.savez(out / 'targets.npz', **targets)
  counts = {}
  for r in results:
    key = f"{r['status']}/{r['reason']}"
    counts[key] = counts.get(key, 0) + 1
  valid = [r for r in results if r['delta_g'] is not None]
  meta = dict(
      policy=str(policy), corpus=str(corpus), collection=str(collection),
      gamma=[CF.GAMMA.numerator, CF.GAMMA.denominator], clock='S3_MARKET_MINUTE',
      coupling_floor=reference.floor, reference_rows=reference.n,
      legal_flat_decisions=len(reference.forks()),
      forks=dict(requested_waits=forks, seed=seed, chosen=len(chosen),
                 reference_adds=int(sum(r['reference_action'] == ADD for r in results))),
      max_steps=max_steps, counts=counts, valid_pairs=len(valid),
      delta_g=dict(mean=float(np.mean([r['delta_g'] for r in valid])) if valid else None,
                   std=float(np.std([r['delta_g'] for r in valid])) if valid else None),
      branch_steps=dict(total=int(sum(r['steps'] for r in results)),
                        max=int(max((r['steps'] for r in results), default=0))),
      replay=dict(steps=[p['replay_steps'] for p in parts], z_differs=[p['z_differs'] for p in parts],
                  seconds=[p['replay_seconds'] for p in parts]),
      worker_seconds=[p['seconds'] for p in parts], seconds=round(time.time() - began, 1),
      inference=dict(platform=parts[0]['platform'], compute_dtype=parts[0]['compute_dtype']),
      labels=dict(q_real='STUDENT_EXECUTED_NATIVE', q_cf='VERIFIED_CF_NATIVE',
                  policy_bound='targets are Q^{pi_k} of this policy only'))
  (out / 'meta.json').write_text(json.dumps(meta, indent=1))
  log(f'CF done in {meta["seconds"]} s: {counts}')
  return meta


def cf_targets(reference, results):
  """Per reference row: q_real (complete own return) and, at forks with a
  valid branch, q_cf (the other action's return)."""
  n = reference.n
  out = dict(cursor=reference.start + np.arange(n), q_real=reference.g.astype(np.float32),
             q_real_valid=np.ones(n, np.float32), q_cf=np.zeros(n, np.float32),
             q_cf_valid=np.zeros(n, np.float32), cf_action=np.full(n, -1, np.int8))
  for r in results:
    i = reference.row(r['cursor'])
    out['cf_action'][i] = r['first_action']
    if r['delta_g'] is not None:
      out['q_cf'][i] = r['g']
      out['q_cf_valid'][i] = 1.
  return out


class QWindows:
  """L4 windows over the reference rows of one collection with its targets.

  split='train' holds rows before the last `holdout`, 'eval' the rest. A
  window starts from the zero market state like the policy at the first
  market minute (its first `burnin` rows carry no Q loss). Windows are
  sampled in proportion to their Q rows after the burn-in, a fork row with
  a valid branch counting `cf_boost` times. Outcome and demonstration
  labels are absent (zero); `action` is the executed pi_k action.
  """

  def __init__(self, collection, targets, split='train', holdout=2048, burnin=0,
               cf_boost=16., seed=0):
    rows, arrays, _ = load_collection(collection)
    t = dict(np.load(targets)) if not isinstance(targets, dict) else targets
    n = len(rows)
    if not np.array_equal(t['cursor'], [r['cursor'] for r in rows]):
      raise ValueError('Q targets and the collection have different cursors')
    if not 0 < holdout < n:
      raise ValueError(f'Holdout of {holdout} rows does not fit {n} rows')
    self.lo, self.hi = {'train': (0, n - holdout), 'eval': (n - holdout, n)}[split]
    self.burnin, self.cf_boost = burnin, cf_boost
    self.rng = np.random.default_rng(seed)
    self.data = {k: arrays[k] for k in WINDOW_OBS}
    # The observation of a row carries the reward of the previous action.
    self.data['reward'] = np.concatenate([[0.], arrays['reward'][:-1]]).astype(np.float32)
    self.data['action'] = arrays['action'].astype(np.int32)
    self.data.update({k: t[k].astype(np.float32) for k in Q_KEYS})
    self._cdf = {}

  def _starts(self, length):
    if length not in self._cdf:
      if self.hi - self.lo < length:
        raise ValueError(f'Window of {length} rows exceeds the split')
      d = self.data
      marks = d['q_real_valid'] + self.cf_boost * d['q_cf_valid']
      marks = np.concatenate([[0.], np.cumsum(marks[self.lo:self.hi])])
      start = np.arange(self.hi - self.lo - length + 1)
      w = marks[start + length] - marks[np.minimum(start + self.burnin, start + length)]
      self._cdf[length] = np.cumsum(w)
    return self._cdf[length]

  def _stack(self, starts, length):
    out = {k: np.stack([v[s:s + length] for s in starts]) for k, v in self.data.items()}
    shape = out['action'].shape
    out['is_first'] = np.zeros(shape, bool)
    out['is_first'][:, 0] = True
    out['is_last'] = np.zeros(shape, bool)
    out['is_terminal'] = np.zeros(shape, bool)
    out['account_valid'] = np.ones(shape, np.float32)
    for k, s in {**L.LABEL_SPACE, **L.DEMO_SPACE}.items():
      out[k] = np.zeros(shape + tuple(s), np.float32)
    return out

  def sample(self, batch, length):
    cdf = self._starts(length)
    picks = np.searchsorted(cdf, self.rng.random(batch) * cdf[-1], side='right')
    return self._stack([self.lo + int(i) for i in picks], length)

  def batches(self, batch, length):
    """Every non-overlapping window of the split, in full batches."""
    starts = list(range(self.lo, self.hi - length + 1, length))
    for i in range(0, len(starts) - batch + 1, batch):
      yield self._stack(starts[i:i + batch], length)

  def constant(self):
    """Mean train targets: the state-free answer to beat."""
    d, lo, hi = self.data, self.lo, self.hi
    real, cf = d['q_real_valid'][lo:hi] > 0, d['q_cf_valid'][lo:hi] > 0
    dg = np.where(d['action'][lo:hi] == ADD, d['q_real'][lo:hi] - d['q_cf'][lo:hi],
                  d['q_cf'][lo:hi] - d['q_real'][lo:hi])
    return dict(q_real=float(d['q_real'][lo:hi][real].mean()),
                q_cf=float(d['q_cf'][lo:hi][cf].mean()) if cf.any() else 0.,
                delta_g=float(dg[cf].mean()) if cf.any() else 0.)

  def kappa(self):
    """std(Delta G) of the valid pairs of the split (A5 kappa_Q)."""
    d, lo, hi = self.data, self.lo, self.hi
    cf = d['q_cf_valid'][lo:hi] > 0
    dg = np.where(d['action'][lo:hi] == ADD, d['q_real'][lo:hi] - d['q_cf'][lo:hi],
                  d['q_cf'][lo:hi] - d['q_real'][lo:hi])[cf]
    return float(dg.std()), int(cf.sum())
