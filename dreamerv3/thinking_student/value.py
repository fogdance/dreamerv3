"""S3: policy-conditioned Q teaching material of a frozen complete student pi_k
(C3 5.5, A4, A6 2.4).

reference  pi_k's formal TRAIN collection (run.collect): one decision per
           market cursor from the first minute to the true terminal, as a gym
           ts_v1_cf.Reference with its complete discounted Native returns.
cf         same-state ADD/WAIT branches from legal flat decisions of the
           reference, continued by the same frozen pi_k on its recorded device
           and dtype; coupled branches get exact returns (ts_v1_cf: EXACT
           where the branch's Native account and the policy's own memory
           equal the reference's, FUNDS where enough equity keeps cash out of
           every later decision).
targets    per reference row: q_real, the executed action's complete Native
           return (STUDENT_EXECUTED_NATIVE); per fork row: q_cf, the other
           action's branch return (VERIFIED_CF_NATIVE, VALID_COMPLETE only).
QWindows   L4 training windows over the reference rows; MultiQWindows mixes
           the targets of one pi_k on several TRAIN contracts (C4 3.1).

Every CF worker first replays the whole reference in its own process and
refuses any action or reward that differs from the collection: the branches
and the coupling witness then compare against decisions of the same
compiled policy. The replay records, at each flat decision, the Native
state and the policy's own memory that an EXACT coupling compares. The
targets belong to pi_k only (A6 2.4): a new policy needs its own
collection and CF.

With a budget, of the forks the reference cannot decide (no FUNDS coupling
and no true terminal within the cap, ts_v1_cf Budget) a sample stratified
by what is known at the decision runs to the cap, and every other one runs
a probe: an EXACT coupling or the branch's own terminal within the probe
gives the exact label, a branch still open after it is UNAVAILABLE_BUDGET
and carries no label.

  python -m dreamerv3.thinking_student.run cf --policy DIR --corpus DIR [--forks N]
"""
import hashlib
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


def build_reference(folder, env, floor=None):
  """The collection as a ts_v1_cf.Reference (complete only when the
  collection ended at the true terminal); the coupling floor of the
  contract unless `floor` gives the one of the CF."""
  rows, arrays, run = load_collection(folder)
  terminal = bool(run['done'] and run['env']['terminated'])
  return CF.Reference(
      env.market_clock_counter, [r['cursor'] for r in rows], arrays['reward'],
      [r['equity'] for r in rows], {k: arrays[k] for k in CF.INPUTS},
      arrays['action'], terminal, CF.coupling_floor(env) if floor is None else floor), rows


def choose_forks(reference, forks=None, seed=0):
  """Legal flat decisions of the reference: every pi_k ADD and all WAITs, or
  `forks` WAITs sampled uniformly (never by outcome, C3 5.5)."""
  cursors = reference.forks()
  own = reference.action[cursors - reference.start]
  adds, waits = cursors[own == ADD], cursors[own == WAIT]
  if forks is not None and forks < len(waits):
    waits = np.sort(np.random.default_rng(seed).choice(waits, forks, replace=False))
  return sorted(int(t) for t in np.concatenate([adds, waits]))


def undecided_sample(reference, forks, budget, close, seed=0, window=60):
  """At most `budget` of the forks the reference cannot decide, to run to
  the cap, drawn systematically over strata known at the decision, never by
  outcome (the others get a probe): the
  reference action (behavior), the quartile of its equity (account risk) and
  the half of the volatility of the last `window` closes (market state).
  Returns the sorted sample and the strata with their sizes."""
  forks = np.asarray(sorted(forks), np.int64)
  if not len(forks):
    return [], dict(strata={})
  rows = forks - reference.start
  action, equity = reference.action[rows], reference.equity[rows]
  log_close = np.log(np.asarray(close, np.float64))
  vol = np.array([np.std(np.diff(log_close[max(0, t - window):t + 1])) if t else 0.
                  for t in forks])
  edges, median = np.quantile(equity, [.25, .5, .75]), float(np.median(vol))
  risk, high = np.searchsorted(edges, equity, side='right'), (vol > median).astype(np.int64)
  order = np.lexsort((forks, high, risk, action))
  n = min(int(budget), len(forks))
  step = len(forks) / max(n, 1)
  start = np.random.default_rng([int(seed), 2]).uniform(0, step)
  taken = np.zeros(len(forks), bool)
  taken[order[np.floor(start + step * np.arange(n)).astype(np.int64)]] = True
  strata = {}
  for a, q, h, took in zip(action, risk, high, taken):
    s = strata.setdefault(f'{("WAIT", "ADD")[a]}/equity_q{q + 1}/vol_{("low", "high")[h]}',
                          dict(forks=0, simulated=0))
    s['forks'], s['simulated'] = s['forks'] + 1, s['simulated'] + int(took)
  return [int(t) for t in forks[taken]], dict(
      strata=dict(sorted(strata.items())), equity_quartiles=[float(e) for e in edges],
      vol_window=window, vol_median=median)


def policy_memory(agent, carry):
  """What the frozen policy carries from one decision to the next besides
  the observation: its seed counter and (a digest of the bytes of) its
  recurrent state."""
  import jax
  digest = hashlib.blake2b(digest_size=16)
  for leaf in jax.tree.leaves(jax.device_get(carry)):
    leaf = np.asarray(leaf)
    digest.update(f'{leaf.dtype}{leaf.shape}'.encode())
    digest.update(leaf.tobytes())
  return int(agent.n_actions), digest.hexdigest()


class Stepper:
  """ts_v1_cf stepper over run.Autonomous (the frozen pi_k on a Native env)."""

  def __init__(self, run):
    self.run, self.first_trade = run, len(run.env.attributions)
    self.status = None  # the Native status after the last step (MARGIN_CALL, MAIN_ROLL, ...)

  def state(self):
    env = self.run.env
    return dict(cursor=int(env.current_step), equity=float(env._calculate_equity()),
                **{k: self.run.obs[k] for k in CF.INPUTS})

  def step(self, action=None):
    row, arrays = self.run.step(action)
    self.status = row['status']
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

  def native(self):
    """The branch account's ts_v1_cf.native_state (EXACT coupling)."""
    return CF.native_state(self.run.env)

  def memory(self):
    """policy_memory of the branch before its next decision."""
    return policy_memory(self.run.agent, self.run.carry)

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
  reference, recorded = build_reference(job['collection'], env, job['floor'])
  forks, probe, saved, own, memory = set(job['forks']), set(job['probe']), {}, {}, {}
  main = Autonomous(agent, env, 'policy')
  z_differs = 0
  while not main.done:
    t = int(env.current_step)
    if t in forks:
      saved[t] = main.state()
    if reference.flat[t - reference.start]:
      # What an EXACT coupling at t compares: the Native account and the
      # policy's own memory before its decision.
      reference.record(t, CF.native_state(env))
      memory[t] = policy_memory(agent, main.carry)
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
    result = CF.run_branch(
        stepper, reference, first, job['max_steps'],
        witness=lambda c: (stepper.peek(), stepper.memory()) == (own[c], memory[c]),
        probe=job['probe_steps'] if t in probe else None)
    result.update(CF.pair(reference, result), trades=stepper.trades(), end_status=stepper.status)
    out.append(result)
    if job['log_every'] and len(out) % job['log_every'] == 0:
      print(f'worker {job["index"]}: {len(out)}/{len(forks)} branches', flush=True)
  return dict(index=job['index'], results=out, replay_steps=len(own), z_differs=int(z_differs),
              replay_seconds=round(replay_seconds, 1), seconds=round(time.time() - began, 1),
              platform=agent.config.jax.platform, compute_dtype=agent.config.jax.compute_dtype)


def run_cf(policy, corpus, collection, out, forks=None, seed=0, workers=4,
           max_steps=8192, log_every=50, log=print, budget=None, probe_steps=512):
  """Write out/pairs.jsonl, out/targets.npz and out/meta.json. With
  `budget`, at most that many of the forks the reference cannot decide run
  to the cap (undecided_sample), the others for `probe_steps` steps; those
  still open then are UNAVAILABLE_BUDGET."""
  from .run import corpus_env
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  began = time.time()
  env = corpus_env(corpus)
  reference, _ = build_reference(collection, env)
  if not reference.complete:
    raise ValueError('The collection did not reach the true terminal: no complete returns')
  if budget is not None and budget < 0:
    raise ValueError(f'A CF budget counts forks, >= 0; got {budget}')
  if budget is not None and not 0 < probe_steps < max_steps:
    raise ValueError(f'A CF probe runs 0 < steps < max_steps ({max_steps}); got {probe_steps}')
  chosen = choose_forks(reference, forks, seed)
  undecided = [t for t in chosen if not reference.decidable(t, max_steps)]
  full, strata = undecided, None
  if budget is not None:
    full, strata = undecided_sample(reference, undecided, budget, env._raw.Close.to_numpy(), seed)
  probed = set(undecided) - set(full)
  groups = [chosen[i::workers] for i in range(workers)]
  jobs = [dict(index=i, policy=str(policy), corpus=str(corpus), collection=str(collection),
               floor=reference.floor, forks=g, probe=[t for t in g if t in probed],
               probe_steps=probe_steps, max_steps=max_steps, log_every=log_every)
          for i, g in enumerate(groups) if g]
  log(f'CF: {len(chosen)} forks of {len(reference.forks())} legal flat decisions, '
      f'{len(jobs)} workers')
  if undecided:
    log(f'CF: {len(undecided)} forks the reference cannot decide within {max_steps} steps '
        f'(no FUNDS coupling, no true terminal): {len(full)} run to the cap, {len(probed)} '
        f'probed for {probe_steps} steps')
  parts = []
  if jobs:
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
                 reference_adds=int(sum(r['reference_action'] == ADD for r in results)),
                 undecided=len(undecided)),
      couple_from=reference.couple_from,
      budget=None if budget is None else dict(
          forks=budget, undecided=len(undecided), full=len(full), probed=len(probed),
          probe_steps=probe_steps,
          probe_closed=sum(r['cursor'] in probed and r['status'] != 'UNAVAILABLE_BUDGET'
                           for r in results),
          unavailable=sum(r['status'] == 'UNAVAILABLE_BUDGET' for r in results), **strata),
      coupled={k: sum(r['coupling'] == k for r in results) for k in ('EXACT', 'FUNDS')},
      end_status={s: sum(r['end_status'] == s for r in results)
                  for s in sorted({r['end_status'] for r in results} - {None})},
      max_steps=max_steps, counts=counts, valid_pairs=len(valid),
      delta_g=dict(mean=float(np.mean([r['delta_g'] for r in valid])) if valid else None,
                   std=float(np.std([r['delta_g'] for r in valid])) if valid else None),
      branch_steps=dict(total=int(sum(r['steps'] for r in results)),
                        max=int(max((r['steps'] for r in results), default=0))),
      replay=dict(steps=[p['replay_steps'] for p in parts], z_differs=[p['z_differs'] for p in parts],
                  seconds=[p['replay_seconds'] for p in parts]),
      worker_seconds=[p['seconds'] for p in parts], seconds=round(time.time() - began, 1),
      inference=dict(platform=parts[0]['platform'], compute_dtype=parts[0]['compute_dtype'])
      if parts else None,
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

  split='train' holds rows before the last `holdout` (every row with
  holdout=0), 'eval' the rest. A
  window starts from the zero market state like the policy at the first
  market minute (its first `burnin` rows carry no Q loss), or, given the
  contract's MarketCarry `carry`, from the stored causal carry at its first
  cursor ('carry/*'). Windows are
  sampled in proportion to their Q rows after the burn-in, a fork row with
  a valid branch counting `cf_boost` times. Outcome and demonstration
  labels are absent (zero); `action` is the executed pi_k action.
  `remedial` lists the cursors of decision errors (S5 feedback, valid pairs
  only): their rows count `remedial_boost` times more; every other row is
  sampled as before. `curriculum()` counts what the windows showed.
  """

  def __init__(self, collection, targets, split='train', holdout=2048, burnin=0,
               cf_boost=16., seed=0, carry=None, remedial=None, remedial_boost=1.):
    rows, arrays, _ = load_collection(collection)
    t = dict(np.load(targets)) if not isinstance(targets, dict) else targets
    n = len(rows)
    if not np.array_equal(t['cursor'], [r['cursor'] for r in rows]):
      raise ValueError('Q targets and the collection have different cursors')
    if not 0 <= holdout < n or (split == 'eval' and not holdout):
      raise ValueError(f'Holdout of {holdout} rows does not fit {n} rows for split {split}')
    self.lo, self.hi = {'train': (0, n - holdout), 'eval': (n - holdout, n)}[split]
    self.burnin, self.cf_boost = burnin, cf_boost
    self.rng = np.random.default_rng(seed)
    self.data = {k: arrays[k] for k in WINDOW_OBS}
    # The observation of a row carries the reward of the previous action.
    self.data['reward'] = np.concatenate([[0.], arrays['reward'][:-1]]).astype(np.float32)
    self.data['action'] = arrays['action'].astype(np.int32)
    self.data.update({k: t[k].astype(np.float32) for k in Q_KEYS})
    self.cursor, self.carry, self._cdf = np.asarray(t['cursor']), carry, {}
    self.error, self.remedial_boost = np.zeros(n, bool), float(remedial_boost)
    if remedial is not None:
      at = np.searchsorted(self.cursor, remedial)
      if (at >= n).any() or not np.array_equal(self.cursor[np.minimum(at, n - 1)], remedial):
        raise ValueError('Remedial cursors outside the collection')
      if not (self.data['q_cf_valid'][at] > 0).all():
        raise ValueError('A decision error needs a valid ADD/WAIT pair at its row')
      self.error[at] = True
    self.boost = np.where(self.error, self.remedial_boost, 1.)
    self.seen = np.zeros(n, bool)
    self.windows = self.remedial_windows = 0

  def _starts(self, length):
    if length not in self._cdf:
      if self.hi - self.lo < length:
        raise ValueError(f'Window of {length} rows exceeds the split')
      d = self.data
      marks = (d['q_real_valid'] + self.cf_boost * d['q_cf_valid']) * self.boost
      marks = np.concatenate([[0.], np.cumsum(marks[self.lo:self.hi])])
      start = np.arange(self.hi - self.lo - length + 1)
      w = marks[start + length] - marks[np.minimum(start + self.burnin, start + length)]
      self._cdf[length] = np.cumsum(w)
    return self._cdf[length]

  def _stack(self, starts, length):
    out = {k: np.stack([v[s:s + length] for s in starts]) for k, v in self.data.items()}
    shape = out['action'].shape
    out['is_first'] = np.zeros(shape, bool)
    if self.carry is None:
      out['is_first'][:, 0] = True
    else:
      # Row 0 of a collection is the first market minute.
      out['is_first'][:, 0] = np.asarray(starts) == 0
      out.update(self.carry.at(self.cursor[np.asarray(starts)]))
    out['is_last'] = np.zeros(shape, bool)
    out['is_terminal'] = np.zeros(shape, bool)
    out['account_valid'] = np.ones(shape, np.float32)
    for k, s in {**L.LABEL_SPACE, **L.DEMO_SPACE}.items():
      out[k] = np.zeros(shape + tuple(s), np.float32)
    return out

  def _pick(self, batch, length):
    cdf = self._starts(length)
    picks = np.searchsorted(cdf, self.rng.random(batch) * cdf[-1], side='right')
    return [self.lo + int(i) for i in picks]

  def _count(self, starts, length):
    for s in starts:
      shown = slice(s + self.burnin, s + length)
      self.seen[shown] = True
      self.windows += 1
      self.remedial_windows += int(self.error[shown].any())

  def sample(self, batch, length):
    starts = self._pick(batch, length)
    self._count(starts, length)
    return self._stack(starts, length)

  def curriculum(self):
    """Q rows the sampled windows showed after the burn-in: distinct rows
    and remediation exposure (windows that hold a decision error). Repeats
    of a row are optimizer exposure, never new market samples (EDU-R10)."""
    d, seen, error = self.data, self.seen, self.error
    fork = d['q_cf_valid'] > 0
    return dict(windows=self.windows, remedial_windows=self.remedial_windows,
                remedial_boost=self.remedial_boost, remedial_rows=int(error.sum()),
                remedial_rows_seen=int((error & seen).sum()),
                unique_q_rows=int((seen & (d['q_real_valid'] > 0)).sum()),
                unique_pairs=int((seen & fork).sum()))

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

  def pairs(self):
    """Delta G = G(ADD) - G(WAIT) of every valid pair of the split."""
    d, lo, hi = self.data, self.lo, self.hi
    cf = d['q_cf_valid'][lo:hi] > 0
    return np.where(d['action'][lo:hi] == ADD, d['q_real'][lo:hi] - d['q_cf'][lo:hi],
                    d['q_cf'][lo:hi] - d['q_real'][lo:hi])[cf]

  def kappa(self):
    """std(Delta G) of the valid pairs of the split (A5 kappa_Q)."""
    dg = self.pairs()
    return float(dg.std()), len(dg)


def _contract(meta):
  return json.loads((Path(meta['corpus']) / 'source.json').read_text())['contract']


class MultiQWindows:
  """L4 windows over the CF targets of one frozen pi_k on several TRAIN
  contracts (C4 3.1). Each window first draws its contract (`mix` as in
  corpus.MultiCorpusWindows), then a window of that contract's reference
  rows exactly as QWindows draws it; a window never joins two contracts.
  Every cf folder must hold targets of the same frozen policy: Q targets
  are Q^{pi_k} of their own policy only (A6 2.4). `carry_root` holds the
  contracts' MarketCarry stores: windows then start from the causal carry.
  `remedial` ({contract: decision-error cursors}, S5 feedback of the same
  pi_k) boosts those rows by `remedial_boost` inside their contract; the
  contract mix stays as configured."""

  def __init__(self, folders, split='train', holdout=2048, burnin=0, cf_boost=16.,
               seed=0, mix='uniform_contract', carry_root=None, remedial=None,
               remedial_boost=1.):
    from .corpus import MarketCarry
    self.folders = [Path(f) for f in folders]
    metas = [json.loads((f / 'meta.json').read_text()) for f in self.folders]
    policies = sorted({str(Path(m['policy']).resolve()) for m in metas})
    if len(policies) != 1:
      raise ValueError(f'Q targets of different policies cannot be mixed: {policies}')
    self.policy, self.contracts = policies[0], [_contract(m) for m in metas]
    if len(set(self.contracts)) != len(self.contracts):
      raise ValueError(f'One contract appears twice: {self.contracts}')
    unknown = set(remedial or ()) - set(self.contracts)
    if unknown:
      raise ValueError(f'Remedial cursors of contracts without Q targets: {sorted(unknown)}')
    self.parts = [QWindows(m['collection'], f / 'targets.npz', split, holdout, burnin, cf_boost,
                           carry=carry_root and MarketCarry(Path(carry_root) / c),
                           remedial=(remedial or {}).get(c), remedial_boost=remedial_boost)
                  for f, m, c in zip(self.folders, metas, self.contracts)]
    self.rng = np.random.default_rng(seed)
    for part in self.parts:
      part.rng = self.rng
    w = (np.ones(len(self.parts)) if mix == 'uniform_contract'
         else np.array([p.hi - p.lo for p in self.parts], np.float64))
    self.mix, self.p = mix, w / w.sum()
    self.exposure = dict.fromkeys(self.contracts, 0)

  def sample(self, batch, length):
    which = self.rng.choice(len(self.parts), batch, p=self.p)
    rows = []
    for i in which:
      part = self.parts[i]
      starts = part._pick(1, length)
      part._count(starts, length)
      rows.append(part._stack(starts, length))
      self.exposure[self.contracts[i]] += 1
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}

  def curriculum(self):
    """What the sampled windows showed, per contract and in total (S5 B2)."""
    parts = {c: p.curriculum() for c, p in zip(self.contracts, self.parts)}
    total = {k: sum(v[k] for v in parts.values()) for k in next(iter(parts.values()))
             if k != 'remedial_boost'}
    return dict(total, remedial_boost=self.parts[0].remedial_boost,
                remedial_share=total['remedial_windows'] / max(total['windows'], 1),
                contracts=parts, exposure_note=EXPOSURE_NOTE)

  def batches(self, batch, length):
    """Every non-overlapping window of every contract, in full batches of
    one contract each."""
    for part in self.parts:
      yield from part.batches(batch, length)

  def pair_counts(self):
    counts = {c: len(p.pairs()) for c, p in zip(self.contracts, self.parts)}
    return dict(total=sum(counts.values()), **counts)

  def constant(self):
    """Mean targets over the valid rows of every contract: the state-free
    answer to beat on the pooled rows."""
    real, cf = [], []
    for p in self.parts:
      d, lo, hi = p.data, p.lo, p.hi
      real.append(d['q_real'][lo:hi][d['q_real_valid'][lo:hi] > 0])
      cf.append(d['q_cf'][lo:hi][d['q_cf_valid'][lo:hi] > 0])
    real, cf = np.concatenate(real), np.concatenate(cf)
    dg = np.concatenate([p.pairs() for p in self.parts])
    return dict(q_real=float(real.mean()), q_cf=float(cf.mean()) if len(cf) else 0.,
                delta_g=float(dg.mean()) if len(dg) else 0.)

  def kappa(self, floor=None):
    """kappa_Q (A5) of the valid train pairs of every contract
    (pooled_kappa)."""
    return pooled_kappa([p.pairs() for p in self.parts], self.p, self.contracts,
                        f'std of Delta G over valid train pairs, {self.mix} weights', floor)


EXPOSURE_NOTE = ('remediation repeats TRAIN rows: windows and repeats are optimizer exposure, '
                 'unique rows count distinct market cursors (EDU-R10)')


def pooled_kappa(dgs, p, contracts, statistic, floor=None):
  """kappa_Q (A5): the std of Delta G over the valid pairs `dgs` of every
  contract, a contract's pairs sharing its mix weight `p` (uniform_contract:
  every contract with pairs weighs the same). It needs two valid pairs: a
  floor never stands in for missing pair supervision. A zero std is an
  error unless a positive `floor` is given; the floor also replaces a
  smaller std. Returns (kappa, how it was computed)."""
  n = sum(len(d) for d in dgs)
  std = 0.
  if n:
    x = np.concatenate(dgs)
    w = np.concatenate([np.full(len(d), q / len(d)) for d, q in zip(dgs, p) if len(d)])
    w = w / w.sum()
    std = float(np.sqrt((w * (x - (w * x).sum()) ** 2).sum()))
  source = dict(statistic=statistic, pairs=n, std=std, floor=floor, floored=False,
                contracts={c: dict(pairs=len(d), std=float(d.std()) if len(d) else None)
                           for c, d in zip(contracts, dgs)})
  if n < 2:
    raise ValueError(f'kappa_Q needs two valid ADD/WAIT pairs, the targets hold {n}')
  if not std > 0 or (floor is not None and std < floor):
    if floor is None or not floor > 0:
      raise ValueError(f'kappa_Q is degenerate ({n} valid pairs, std {std}); '
                       'a positive floor is needed')
    source['floored'] = True
    return float(floor), source
  return std, source


def target_pairs(targets):
  """Delta G = G(ADD) - G(WAIT) of every valid pair of a cf_targets table:
  the branch took the other action than pi_k at its fork."""
  t = targets
  dg = np.where(t['cf_action'] == WAIT, t['q_real'] - t['q_cf'], t['q_cf'] - t['q_real'])
  return dg[t['q_cf_valid'] > 0]


def cf_kappa(folders, mix='uniform_contract', floor=None):
  """kappa_Q of every valid pair of the CF folders of one frozen pi_k, read
  from their targets alone (MultiQWindows.kappa with holdout 0, without
  loading the collections). The cf folders and statistic are in the source."""
  folders = [Path(f) for f in folders]
  metas = [json.loads((f / 'meta.json').read_text()) for f in folders]
  if len({str(Path(m['policy']).resolve()) for m in metas}) != 1:
    raise ValueError('kappa_Q pools the CF of one frozen policy only')
  targets = [dict(np.load(f / 'targets.npz')) for f in folders]
  w = (np.ones(len(folders)) if mix == 'uniform_contract'
       else np.array([len(t['cursor']) for t in targets], np.float64))
  kappa, source = pooled_kappa([target_pairs(t) for t in targets], w / w.sum(),
                               [_contract(m) for m in metas],
                               f'std of Delta G over every valid pair, {mix} weights', floor)
  return kappa, dict(source, cf=[str(f) for f in folders])


CHECK_KEYS = ('delta_q', 'z', 'p_tp_h', 'correction', 'q_add', 'q_wait')
# Distance of the continuous z from the decision boundary z = 0, in units of
# the std of z on the legal rows after the burn-in.
BOUNDARY_EDGES = (0., .05, .1, .25, .5, 1., np.inf)


def position_bins(length):
  """Positions in a window: 16-row bins up to row 64, then doubling bins."""
  edges, b = [a for a in (0, 16, 32, 48) if a < length], 64
  while b < length:
    edges.append(b)
    b *= 2
  return list(zip(edges, edges[1:] + [length]))


def path_agreement(short, cont, act_short, act_cont, legal, burnin, keys=CHECK_KEYS):
  """Window answers against the continuous path at the same cursors (arrays
  [window, position, ...]): mean |difference| by position, also in units of
  the continuous std after the burn-in, the mean signed difference (window
  minus path, over every row: a bias, which the rows a path's own ADDs mask
  cannot select), and flips of the formal masked-argmax action on the rows
  whose mask allows ADD. A masked row is WAIT on both paths and would only
  dilute a flip rate, so it never enters one. The flips after the burn-in
  are also split by how far the continuous z lies from the decision
  boundary z = 0 (BOUNDARY_EDGES)."""
  pos = np.arange(short.shape[1])[None, :]
  signed, flips = short - cont, act_short != act_cont
  diff, post = np.abs(signed), legal & (pos >= burnin)

  def in_std(value, std):
    return float(value / std) if std > 0 else None

  def flip_rate(rows):
    n = int(rows.sum())
    return dict(legal_rows=n, flips=int(flips[rows].sum()),
                flip_rate=float(flips[rows].mean()) if n else None)

  std = {k: float(cont[:, burnin:, i].std()) for i, k in enumerate(keys)}
  by_position = {}
  for a, b in position_bins(short.shape[1]):
    d = {k: float(diff[:, a:b, i].mean()) for i, k in enumerate(keys)}
    m = {k: float(signed[:, a:b, i].mean()) for i, k in enumerate(keys)}
    by_position[f'{a}-{b}'] = dict(
        mean_abs_diff=d, in_std={k: in_std(v, std[k]) for k, v in d.items()},
        mean_diff=m, mean_diff_in_std={k: in_std(v, std[k]) for k, v in m.items()},
        **flip_rate(legal & (pos >= a) & (pos < b)))
  after_burnin = {}
  for i, k in enumerate(keys):
    d, m, legal_c = diff[:, burnin:, i], signed[:, burnin:, i].mean(), cont[..., i][post]
    after_burnin[k] = dict(
        mean_abs_diff=float(d.mean()), max_abs_diff=float(d.max()), continuous_std=std[k],
        in_std=in_std(d.mean(), std[k]), mean_diff=float(m), mean_diff_in_std=in_std(m, std[k]),
        legal_in_std=in_std(diff[..., i][post].mean(), legal_c.std()) if post.any() else None)
  z = cont[..., keys.index('z')][post]
  z_std = float(z.std()) if post.any() else 0.
  dist, f = (np.abs(z) / z_std if z_std > 0 else np.zeros_like(z)), flips[post]
  bins = {}
  for lo, hi in zip(BOUNDARY_EDGES[:-1], BOUNDARY_EDGES[1:]):
    near = (dist >= lo) & (dist < hi)
    bins[f'{lo:g}-{hi:g}'] = dict(legal_rows=int(near.sum()), flips=int(f[near].sum()),
                                  flip_rate=float(f[near].mean()) if near.any() else None)
  a_c, a_s = act_cont[post], act_short[post]
  return dict(
      by_position=by_position, after_burnin=after_burnin,
      actions=dict(**flip_rate(post), continuous_adds=int((a_c == ADD).sum()),
                   window_adds=int((a_s == ADD).sum()),
                   add_to_wait=int(((a_c == ADD) & (a_s == WAIT)).sum()),
                   wait_to_add=int(((a_c == WAIT) & (a_s == ADD)).sum()),
                   masked_flips=int(flips[~legal].sum())),
      boundary=dict(
          unit='|z| of the continuous path / std of z on the legal rows after the burn-in',
          z_std=z_std, bins=bins,
          flip_distance=dict(median=float(np.median(dist[f])), p90=float(np.quantile(dist[f], .9)),
                             max=float(dist[f].max())) if f.any() else None))


def window_check(policy, collection, windows=32, length=64, seed=0, device=None, start='zero',
                 seed_offset=0, store=None, log=print):
  """RSSM short-window check (C4 3.1). The frozen policy reads the decision
  inputs of a formal collection once continuously from the first market
  minute (its answers and actions as on a continuous path) and again in
  `windows` windows. start='zero': each window starts from the zero market
  state with the loss burn-in, as zero-carry training windows do;
  start='carry': from the market carry before its first row, of the
  continuous path itself or of the contract's MarketCarry folder `store`
  (as carry training windows read it, then compared with the path's carry),
  with no burn-in. With seed_offset 0 a carry window repeats the path
  exactly; with another offset only the stochastic latents of other policy
  seeds (as training seeds are) can move it. Answers and formal actions of
  both at the same cursors are compared by position in the window
  (path_agreement); the window starting at the first minute is the control
  and agrees exactly with seed_offset 0. A diagnostic, never a gate."""
  from .corpus import CARRY_KEYS, MarketCarry
  from .run import DECISION_OBS, load_policy
  if start not in ('zero', 'carry') or (store and start != 'carry'):
    raise ValueError(f"start must be 'zero' or 'carry' (a store needs 'carry'), got {start!r}")
  rows, arrays, _ = load_collection(collection)
  agent, meta = load_policy(policy, device)
  burnin, n, base = int(agent.config.ts.q_burnin), len(rows), int(agent.n_actions)
  burnin = 0 if start == 'carry' else burnin
  store = store and MarketCarry(store)
  if n <= 2 * length:
    raise ValueError(f'{n} collection rows are too few for {length}-row windows')
  reward = np.concatenate([[0.], arrays['reward'][:-1]]).astype(np.float32)
  keys = CHECK_KEYS

  def step(carry, t, first):
    obs = {k: arrays[k][t][None] for k in DECISION_OBS}
    obs.update(reward=reward[t:t + 1], is_first=np.array([first]),
               is_last=np.array([False]), is_terminal=np.array([False]))
    carry, act, out = agent.policy(carry, obs, mode='policy')
    return carry, [float(out[f'answer/{k}'][0]) for k in keys], int(act['action'][0])

  rng = np.random.default_rng(seed)
  starts = [0] + sorted(int(s) for s in rng.choice(np.arange(1, n - length), windows, replace=False))
  began, carry, saved, begins = time.time(), agent.init_policy(1), {}, set(starts)
  path, path_act = np.zeros((n, len(keys))), np.zeros(n, np.int8)
  agent.n_actions.value = base
  for t in range(n):
    if start == 'carry' and t in begins:
      saved[t] = carry
    carry, path[t], path_act[t] = step(carry, t, t == 0)
  gap = None
  if store:
    # The stored carry against the path's at every window start, then the
    # windows start from the store as training windows do.
    at = {s: store.at([rows[s]['cursor']]) for s in starts}
    gap = max(float(np.abs(np.asarray(saved[s]['market'][k][0], np.float32)
                           - at[s][f'carry/{k}'][0].astype(np.float32)).max())
              for s in starts for k in CARRY_KEYS)
    saved = {s: {'market': {k: [at[s][f'carry/{k}'][0]] for k in CARRY_KEYS}} for s in starts}
  short = np.zeros((len(starts), length, len(keys)))
  short_act = np.zeros((len(starts), length), np.int8)
  for w, s in enumerate(starts):
    carry = saved.pop(s) if start == 'carry' else agent.init_policy(1)
    agent.n_actions.value = base + s + seed_offset
    for j in range(length):
      first = j == 0 if start == 'zero' else s + j == 0
      carry, short[w, j], short_act[w, j] = step(carry, s + j, first)
  cont = np.stack([path[s:s + length] for s in starts])
  cont_act = np.stack([path_act[s:s + length] for s in starts])
  mask = np.array([r['mask'][1] > 0.5 for r in rows])
  legal = np.stack([mask[s:s + length] for s in starts])
  dq = keys.index('delta_q')
  dq_s, dq_c = short[1:, burnin:, dq], cont[1:, burnin:, dq]
  signed = legal[1:, burnin:] & (dq_c != 0)
  same = json.loads((Path(collection) / 'meta.json').read_text())['policy'] == meta['policy']
  recorded = np.array([[r[k] for k in keys] for r in rows])
  report = dict(
      policy=str(policy), collection=str(collection), rows=n, windows=len(starts) - 1,
      length=length, burnin=burnin, start=start, seed_offset=seed_offset, keys=list(keys),
      store=store and str(store.folder), store_vs_path_max_abs_diff=gap,
      seconds=round(time.time() - began, 1),
      control_max_abs_diff=float(np.abs(short[0] - cont[0]).max()),
      control_action_flips=int((short_act[0] != cont_act[0]).sum()),
      replay_vs_record_max_abs_diff=float(np.abs(path - recorded).max()) if same else None,
      replay_vs_record_action_flips=(int((path_act != [r['proposed'] for r in rows]).sum())
                                     if same else None),
      **path_agreement(short[1:], cont[1:], short_act[1:], cont_act[1:], legal[1:], burnin, keys),
      delta_q_sign_agree=(float((np.sign(dq_s[signed]) == np.sign(dq_c[signed])).mean())
                          if signed.any() else None),
      delta_q_signed_rows=int(signed.sum()))
  a = report['actions']
  log(f'state-check: control {report["control_max_abs_diff"]:.3g}, after burn-in '
      + ', '.join(f'{k} {v["mean_abs_diff"]:.4g}/{v["continuous_std"]:.4g}'
                  for k, v in report['after_burnin'].items())
      + f'; formal action flips {a["flips"]}/{a["legal_rows"]} legal rows')
  return report
