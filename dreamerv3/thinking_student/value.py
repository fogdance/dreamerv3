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
QWindows   L4 training windows over the reference rows; MultiQWindows mixes
           the targets of one pi_k on several TRAIN contracts (C4 3.1).

The reference is replayed once per contract. Full fork snapshots are spooled
to the output directory (only the unfinished probes keep theirs once the
output is complete); each worker holds one live account, reuses its
environment/policy and reads the compact reference continuation states.
Every candidate gets a short probe. The budget is a continuation quota for
unfinished probes, sampled using decision-time strata. A probe and its
continuation share Native state, policy carry/seeds and one branch cap.
Targets belong to pi_k only (A6 2.4); a new policy needs its own collection/CF.

  python -m dreamerv3.thinking_student.run cf --policy DIR --corpus DIR [--forks N]
"""
import json
import multiprocessing
import os
import pickle
import threading
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


def undecided_sample(reference, forks, budget, close, seed=0, window=60):
  """At most `budget` of the unfinished probes, drawn
  systematically over strata known at the decision, never by outcome: the
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


class Stepper:
  """ts_v1_cf stepper over run.Autonomous (the frozen pi_k on a Native env)."""

  def __init__(self, run):
    self.run, self.first_trade = run, len(run.env.attributions)
    self.status = None  # the Native status after the last step (MARGIN_CALL, MAIN_ROLL, ...)
    self.policy_calls = 0
    self.reference, self.first_joint = None, None
    self.fork_cursor = int(run.env.current_step)

  def continuation(self):
    # StudentModel.policy reads market, is_first and the three account inputs;
    # reward/steps/history are not policy inputs. Its only recurrent state is
    # MarketCarry; the wrapper's n_actions determines all stochastic seeds.
    import jax
    run = self.run
    if run.mode != 'policy':
      raise ValueError('CF requires the frozen formal policy')
    return dict(native=CF.native_continuation(run.env), policy=dict(
        mode=run.mode, first=run.first, done=run.done, n_actions=int(run.agent.n_actions),
        carry=jax.tree.map(lambda x: np.array(x), run.carry)))

  def state(self):
    env = self.run.env
    state = dict(cursor=int(env.current_step), equity=float(env._calculate_equity()),
                 **{k: self.run.obs[k] for k in CF.INPUTS})
    if state['position_state'][1] == 0:
      state.update(self.continuation())
      ref, c = self.reference, state['cursor']
      if (ref is not None and self.first_joint is None and c > self.fork_cursor
          and ref.start <= c < ref.start + ref.n and ref.flat[ref.row(c)]):
        i, saved = ref.row(c), ref.continuations.get(c)
        gap = state['equity'] - ref.equity[i]
        self.first_joint = dict(cursor=c, steps=c - self.fork_cursor,
            equity=state['equity'], reference_equity=float(ref.equity[i]), equity_gap=gap,
            input_differences={k: int(np.count_nonzero(np.asarray(state[k]) != ref.inputs[k][i]))
                               for k in CF.INPUTS},
            funds_safe=bool(ref.future_min_equity[i] + min(gap, 0.) >= ref.floor),
            future_min_equity=float(ref.future_min_equity[i]), coupling_floor=ref.floor,
            native_equal=bool(saved and state['native'] is not None and
                              CF.same_state(state['native'], saved['native'])),
            policy_equal=bool(saved and CF.same_state(state['policy'], saved['policy'])),
            coupling_kind=ref.coupling_kind(state), status=env.status,
            terminated=bool(env.terminated), truncated=bool(env.truncated))
    return state

  def step(self, action=None):
    self.policy_calls += 1
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
    self.policy_calls += 1
    try:
      _, act, out = agent.policy(run.carry, obs, mode=run.mode)
    finally:
      agent.n_actions.value = counter
    return float(out['answer/z'][0]), int(act['action'][0])

  def trades(self):
    keys = ('entry_cursor', 'exit_cursor', 'reason', 'entry_price', 'exit_price',
            'net_trade_pnl')
    return [{k: t.get(k) for k in keys} for t in self.run.env.attributions[self.first_trade:]]


def _pickle_write(path, value):
  """Stream one owned checkpoint to disk; never retain all fork histories."""
  path = Path(path)
  tmp = path.with_suffix('.tmp')
  with tmp.open('wb') as f:
    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
  tmp.replace(path)


def _json_write(path, value):
  path = Path(path)
  tmp = path.with_suffix('.tmp')
  tmp.write_text(json.dumps(value, indent=1))
  tmp.replace(path)


def _cf_init(policy, corpus, work):
  # One agent and one Native account per worker, reused across both phases.
  os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  from .run import corpus_env, load_policy
  global _CF_WORKER
  agent, meta = load_policy(policy, logdir=Path(work) / f'policy_{os.getpid()}')
  _CF_WORKER = dict(agent=agent, env=corpus_env(corpus), meta=meta, prepared=None,
                    n_actions=int(agent.n_actions), inputs=None)


def _replay(main, reference, recorded, forks, work, end=None):
  """The frozen reference from main's cursor to `end` (or the terminal):
  checked against the collection; flat continuations, the policy's own
  answers at flat rows and a checkpoint at each fork."""
  stepper, continuations, own = Stepper(main), {}, {}
  z_differs, save_seconds, save_bytes, steps = 0, 0., 0, 0
  while not main.done and (end is None or int(main.env.current_step) < end):
    t = int(main.env.current_step)
    if reference.flat[reference.row(t)]:
      continuations[t] = stepper.continuation()
    if t in forks:
      started = time.monotonic()
      file = work / f'{t}.pkl'
      _pickle_write(file, dict(run=main.state(), progress=CF.BranchProgress(
          t, 1 - int(reference.action[reference.row(t)])),
          first_trade=len(main.env.attributions), first_joint=None, probe_steps=0,
          phase='fork', cost=dict(native_steps=0, policy_calls=0, seconds=0.,
                                  save_seconds=0., restore_seconds=0.)))
      save_seconds += time.monotonic() - started
      save_bytes += file.stat().st_size
    row, arrays = main.step()
    steps += 1
    rec = recorded[reference.row(t)]
    if (row['cursor'], row['executed'], row['reward']) != (rec['cursor'], rec['executed'], rec['reward']):
      raise RuntimeError(f'The frozen reference replay differs from the collection at cursor {t}')
    if row['equity'] != rec['equity'] or not reference.same_inputs(t, arrays):
      raise RuntimeError(f'The replayed Native account differs from the collection at cursor {t}')
    z_differs += row['z'] != rec['z']
    if reference.flat[reference.row(t)]:
      own[t] = (row['z'], row['executed'])
  return dict(continuations=continuations, own=own, z_differs=z_differs, steps=steps,
              save_seconds=save_seconds, save_bytes=save_bytes,
              terminated=bool(main.env.terminated))


def _cf_segment(job):
  """One segment of the reference replay, from a collection snapshot (or the
  start) to the next snapshot's cursor."""
  from .run import Autonomous
  w, work = _CF_WORKER, Path(job['work'])
  if w['inputs'] is None:
    with (work / 'prepare_inputs.pkl').open('rb') as f:
      w['inputs'] = pickle.load(f)
  reference, recorded, forks = w['inputs']
  began = time.monotonic()
  if job['snapshot'] is None:
    w['agent'].n_actions.value = w['n_actions']
    main = Autonomous(w['agent'], w['env'], 'policy')
  else:
    with open(job['snapshot'], 'rb') as f:
      main = Autonomous.restore(w['agent'], w['env'], pickle.load(f)['run'])
  if int(main.env.current_step) != job['start']:
    raise RuntimeError(f'Snapshot {job["snapshot"]} starts at {main.env.current_step}, not {job["start"]}')
  out = _replay(main, reference, recorded, set(forks), work, job['end'])
  return dict(out, seconds=time.monotonic() - began, platform=w['agent'].config.jax.platform,
              compute_dtype=w['agent'].config.jax.compute_dtype)


def _cf_prepare(job):
  """One reference replay per contract, not per worker or candidate."""
  from .run import Autonomous
  w, work = _CF_WORKER, Path(job['work'])
  path = work / 'reference.pkl'
  if path.exists():
    with path.open('rb') as f:
      w['prepared'] = pickle.load(f)
    return dict(w['prepared']['metrics'], reused=True)
  began = time.monotonic()
  reference, recorded = job['reference'], job['recorded']
  main = Autonomous(w['agent'], w['env'], 'policy')
  out = _replay(main, reference, recorded, set(job['forks']), work)
  if not out['terminated']:
    raise RuntimeError('Reference replay did not reach the Native terminal')
  reference.continuations.update(out['continuations'])
  metrics = dict(native_steps=main.steps, policy_calls=main.steps, z_differs=out['z_differs'],
                 seconds=time.monotonic() - began, snapshot_save_seconds=out['save_seconds'],
                 snapshot_bytes=out['save_bytes'], flat_states=len(reference.continuations),
                 platform=w['agent'].config.jax.platform,
                 compute_dtype=w['agent'].config.jax.compute_dtype)
  w['prepared'] = dict(reference=reference, own=out['own'], metrics=metrics)
  _pickle_write(path, w['prepared'])
  return dict(metrics, reused=False)


def _cf_prepare_parallel(pool, work, reference, recorded, forks, snapshots):
  """The reference replay in segments between the collection's chunk
  snapshots, on every worker; the same reference.pkl as _cf_prepare."""
  path = work / 'reference.pkl'
  if path.exists():
    with path.open('rb') as f:
      return dict(pickle.load(f)['metrics'], reused=True)
  began = time.monotonic()
  starts = []
  for file in snapshots:
    with file.open('rb') as f:
      run = pickle.load(f)['run']
    if not run['done']:
      starts.append((reference.start + int(run['steps']), str(file)))
  bounds = [(reference.start, None), *starts]
  jobs = [dict(work=str(work), start=a, snapshot=snap, end=b[0] if b else None)
          for (a, snap), b in zip(bounds, [*bounds[1:], None])]
  _pickle_write(work / 'prepare_inputs.pkl', (reference, recorded, list(forks)))
  parts = list(pool.map(_cf_segment, jobs))
  if not parts[-1]['terminated']:
    raise RuntimeError('Reference replay did not reach the Native terminal')
  own = {}
  for part in parts:
    reference.continuations.update(part['continuations'])
    own.update(part['own'])
  steps = sum(p['steps'] for p in parts)
  if steps != reference.n:
    raise RuntimeError(f'The replay segments cover {steps} of {reference.n} reference rows')
  metrics = dict(native_steps=steps, policy_calls=steps, z_differs=sum(p['z_differs'] for p in parts),
                 seconds=time.monotonic() - began,
                 snapshot_save_seconds=sum(p['save_seconds'] for p in parts),
                 snapshot_bytes=sum(p['save_bytes'] for p in parts),
                 flat_states=len(reference.continuations), platform=parts[0]['platform'],
                 compute_dtype=parts[0]['compute_dtype'], segments=len(parts),
                 segment_seconds=sum(p['seconds'] for p in parts))
  _pickle_write(path, dict(reference=reference, own=own, metrics=metrics))
  (work / 'prepare_inputs.pkl').unlink()
  return dict(metrics, reused=False)


def _cf_job(job):
  """Probe or continue a batch using the same saved branch state and cap."""
  from .run import Autonomous
  w, work = _CF_WORKER, Path(job['work'])
  if w['prepared'] is None:
    with (work / 'reference.pkl').open('rb') as f:
      w['prepared'] = pickle.load(f)
  reference, own = w['prepared']['reference'], w['prepared']['own']
  out = []
  executed = dict(native_steps=0, policy_calls=0, seconds=0., save_seconds=0., restore_seconds=0.)
  for t in job['forks']:
    final = work / f'{t}.json'
    if final.exists():
      out.append(json.loads(final.read_text()))
      continue
    began = time.monotonic()
    checkpoint = work / f'{t}.pkl'
    with checkpoint.open('rb') as f:
      saved = pickle.load(f)
    progress, cost = saved['progress'], saved['cost']
    probe_record = work / f'{t}.probe.json'
    if saved['phase'] != 'fork' and probe_record.exists():
      saved['record'] = json.loads(probe_record.read_text())
      cost = saved['cost'] = saved['record']['cost']
    # Recovery keeps the advanced probe, including its forced first action.
    if job['phase'] == 'probe' and saved['phase'] != 'fork':
      out.append(saved['record'])
      continue
    before_cost = dict(cost)
    stepper = Stepper(Autonomous.restore(w['agent'], w['env'], saved['run']))
    del saved['run']
    stepper.first_trade = saved['first_trade']
    stepper.first_joint = saved['first_joint']
    stepper.fork_cursor, stepper.reference = t, reference
    cost['restore_seconds'] += time.monotonic() - began
    before = len(progress.rewards)
    stop = min(job['probe_steps'], job['max_steps']) if job['phase'] == 'probe' else job['max_steps']
    result = CF.advance_branch(stepper, reference, progress, stop, job['max_steps'],
                               witness=lambda c: stepper.peek() == own[c])
    cost['native_steps'] += len(progress.rewards) - before
    cost['policy_calls'] += stepper.policy_calls
    if job['phase'] == 'probe':
      saved['probe_steps'] = len(progress.rewards)
    if result is None:
      result = CF.unavailable(reference, t, progress.first_action, 'PROBE_UNFINISHED_BUDGET', progress)
    result.update(CF.pair(reference, result), trades=stepper.trades(),
                  end_status=stepper.run.env.status,
                  native_terminal=bool(stepper.run.env.terminated),
                  native_truncated=bool(stepper.run.env.truncated),
                  first_joint=stepper.first_joint, probe_steps=saved['probe_steps'],
                  continuation_steps=len(progress.rewards) - saved['probe_steps'],
                  completed_in=job['phase'] if result['status'] == 'VALID_COMPLETE' else None)
    pending = result['status'] == 'UNAVAILABLE_BUDGET'
    result['cost'] = cost
    if pending:
      # Full histories go to disk. Only this branch is live in worker RAM.
      started = time.monotonic()
      saved.update(run=stepper.run.state(), first_joint=stepper.first_joint,
                   phase='probe', record=result)
      _pickle_write(checkpoint, saved)
      cost['save_seconds'] += time.monotonic() - started
    cost['seconds'] += time.monotonic() - began
    result['cost'] = dict(cost)
    _json_write(probe_record if pending else final, result)
    for key in executed:
      executed[key] += cost[key] - before_cost[key]
    out.append(result)
    if job['log_every'] and len(out) % job['log_every'] == 0:
      print(f'CF {job["phase"]}: {len(out)}/{len(job["forks"])} branches', flush=True)
  return dict(results=out, executed=executed)


class _PeakMemory:
  """Sample process-tree RSS (shared pages may be counted more than once)."""
  def __enter__(self):
    import psutil
    self.process, self.peak, self.stop = psutil.Process(), 0, threading.Event()
    def sample():
      while not self.stop.is_set():
        total = 0
        for p in [self.process, *self.process.children(recursive=True)]:
          try:
            total += p.memory_info().rss
          except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        self.peak = max(self.peak, total)
        self.stop.wait(.2)
    self.thread = threading.Thread(target=sample, daemon=True)
    self.thread.start()
    return self

  def __exit__(self, *args):
    self.stop.set()
    self.thread.join()


def _base_pairs(folder, policy, corpus, collection, reference):
  """Local supplementation with the same Q semantics; no hash gate."""
  if folder is None:
    return {}
  folder = Path(folder)
  meta = json.loads((folder / 'meta.json').read_text())
  for key, value in dict(policy=policy, corpus=corpus, collection=collection).items():
    if Path(meta[key]).resolve() != Path(value).resolve():
      raise ValueError(f'Base CF {key} differs; cannot mix Q targets')
  if (meta['gamma'] != [CF.GAMMA.numerator, CF.GAMMA.denominator] or
      meta['clock'] != 'S3_MARKET_MINUTE' or meta['reference_rows'] != reference.n):
    raise ValueError('Base CF discount/clock/reference differs')
  rows = [json.loads(line) for line in (folder / 'pairs.jsonl').open()]
  legal = set(reference.forks().tolist())
  if len({r['cursor'] for r in rows}) != len(rows):
    raise ValueError('Duplicate base CF cursor')
  for r in rows:
    if (r['cursor'] not in legal or r['first_action'] != 1 - reference.action[reference.row(r['cursor'])]
        or r['reference_action'] != reference.action[reference.row(r['cursor'])]):
      raise ValueError('Base CF is not a legal opposite-action fork')
    if r['status'] == 'VALID_COMPLETE':
      if not all(r.get(k) is not None and np.isfinite(r[k]) for k in ('g', 'g_add', 'g_wait', 'delta_g')):
        raise ValueError('Base VALID_COMPLETE has no finite pair')
      expected = CF.pair(reference, r)
      if any(not np.isclose(r[k], expected[k], rtol=0, atol=1e-12)
             for k in ('g_add', 'g_wait', 'delta_g')):
        raise ValueError('Base CF pair does not match the reference return')
    elif r['g'] is not None or r['delta_g'] is not None:
      raise ValueError('Incomplete base CF must not contain a numeric Q pair')
  return {r['cursor']: r for r in rows}


def run_cf(policy, corpus, collection, out, forks=None, seed=0, workers=4,
           max_steps=8192, log_every=50, log=print, budget=None, probe_steps=64,
           cursors=None, base_cf=None):
  """Probe every candidate, then continue a stratified quota from saved state.
  A larger quota can resume the same output. Partial supplements use a new
  output and base_cf; completed old Q labels are retained.
  """
  with _PeakMemory() as memory:
    return _run_cf(policy, corpus, collection, out, forks, seed, workers, max_steps,
                   log_every, log, budget, probe_steps, cursors, base_cf, memory)


def _run_cf(policy, corpus, collection, out, forks=None, seed=0, workers=4,
            max_steps=8192, log_every=50, log=print, budget=None, probe_steps=64,
            cursors=None, base_cf=None, memory=None):
  """Probe all candidates, then sample unfinished branches for continuation.
  Work files retain the shared reference, one checkpoint per fork and atomic
  results; a finished fork's checkpoint is deleted once the output is
  complete. Rerun the same command to recover completed probes/results. Use a
  new output + base_cf to supplement an older dataset without overwriting it.
  """
  from .run import corpus_env
  if workers < 1 or max_steps < 1 or not 1 <= probe_steps <= 64:
    raise ValueError('workers/max_steps must be positive and probe_steps in [1, 64]')
  if budget is not None and budget < 0:
    raise ValueError('A CF continuation budget counts forks, >= 0')
  if cursors is not None and forks is not None:
    raise ValueError('Use explicit cursors or WAIT sampling, not both')
  out = Path(out).resolve()
  if base_cf is not None and out == Path(base_cf).resolve():
    raise ValueError('Supplement CF into a new output; base CF is read-only')
  began = time.monotonic()
  env = corpus_env(corpus)
  reference, recorded = build_reference(collection, env)
  if not reference.complete:
    raise ValueError('The collection did not reach the true terminal: no complete returns')
  chosen = choose_forks(reference, forks, seed) if cursors is None else sorted(set(map(int, cursors)))
  if set(chosen) - set(reference.forks().tolist()):
    raise ValueError('Requested cursor is not a legal flat fork')
  inherited = _base_pairs(base_cf, policy, corpus, collection, reference)
  plan = dict(policy=str(Path(policy).resolve()), corpus=str(Path(corpus).resolve()),
              collection=str(Path(collection).resolve()), cursors=chosen,
              max_steps=max_steps, probe_steps=probe_steps, seed=seed, budget=budget,
              base_cf=str(Path(base_cf).resolve()) if base_cf else None)
  work = out / 'work'
  manifest = work / 'plan.json'
  if not manifest.exists() and (out / 'meta.json').exists():
    raise ValueError('Existing CF output is read-only; supplement into a new directory')
  if manifest.exists():
    previous = json.loads(manifest.read_text())
    old_budget = previous.pop('budget')
    if previous != {k: v for k, v in plan.items() if k != 'budget'}:
      raise ValueError('Existing CF work uses another plan; choose a new output directory')
    if budget is not None and (old_budget is None or budget < old_budget):
      raise ValueError('Cannot reduce an already allocated continuation quota')
  work.mkdir(parents=True, exist_ok=True)
  _json_write(manifest, plan)
  close = env._raw.Close.to_numpy()
  del env
  # One fork per job: the pool hands the next fork to whichever worker is free.
  jobs = lambda phase, ts: [dict(work=str(work), phase=phase, forks=[t], probe_steps=probe_steps,
                                 max_steps=max_steps, log_every=log_every) for t in ts]
  log(f'CF: {len(chosen)} legal candidates, probe <= {min(probe_steps, max_steps)} steps each; '
      f'continuation quota {budget}; one shared reference replay')
  reference_metrics, results, sample, survivors, strata = {}, [], [], [], dict(strata={})
  executed = dict(native_steps=0, policy_calls=0, seconds=0., save_seconds=0., restore_seconds=0.)
  if chosen:
    with ProcessPoolExecutor(min(workers, len(chosen)), mp_context=multiprocessing.get_context('spawn'),
                             initializer=_cf_init, initargs=(str(policy), str(corpus), str(work))) as pool:
      snapshots = sorted((Path(collection) / 'snapshots').glob('*.pkl'))
      reference_metrics = (
          _cf_prepare_parallel(pool, work, reference, recorded, chosen, snapshots) if snapshots else
          pool.submit(_cf_prepare, dict(work=str(work), reference=reference,
                                        recorded=recorded, forks=chosen)).result())
      parts = list(pool.map(_cf_job, jobs('probe', chosen)))
      results = [r for part in parts for r in part['results']]
      # Recovered final continuations have already used quota slots. Sample
      # the same original survivor set, including those completed branches.
      survivors = sorted(r['cursor'] for r in results if r['status'] == 'UNAVAILABLE_BUDGET'
                         or r.get('continuation_steps', 0))
      allocation_path = work / 'continuations.json'
      allocation = json.loads(allocation_path.read_text()) if allocation_path.exists() else dict(
          cursors=[], draws=[])
      previous = allocation['cursors']
      if set(previous) - set(survivors):
        raise ValueError('Saved continuation allocation differs from probe survivors')
      quota = len(survivors) if budget is None else min(budget, len(survivors))
      extra, draw = undecided_sample(reference, sorted(set(survivors) - set(previous)),
                                     quota - len(previous), close, seed)
      sample = sorted(previous + extra)
      if extra:
        allocation['draws'].append(draw)
      allocation['cursors'] = sample
      _json_write(allocation_path, allocation)
      strata = dict(draws=allocation['draws'])
      if sample:
        continued = list(pool.map(_cf_job, jobs('continuation', sample)))
        parts.extend(continued)
        by_cursor = {r['cursor']: r for r in results}
        by_cursor.update((r['cursor'], r) for part in continued for r in part['results'])
        results = [by_cursor[t] for t in chosen]
      for part in parts:
        for key in executed:
          executed[key] += part['executed'][key]
  computed = sorted(results, key=lambda r: r['cursor'])
  for r in computed:
    old = inherited.get(r['cursor'])
    if not old or old['status'] != 'VALID_COMPLETE' or r['status'] == 'VALID_COMPLETE':
      inherited[r['cursor']] = r
  results = sorted(inherited.values(), key=lambda r: r['cursor'])
  tmp = out / 'pairs.jsonl.tmp'
  with tmp.open('w') as f:
    f.writelines(json.dumps(r) + '\n' for r in results)
  tmp.replace(out / 'pairs.jsonl')
  with (out / 'targets.tmp').open('wb') as f:
    np.savez(f, **cf_targets(reference, results))
  (out / 'targets.tmp').replace(out / 'targets.npz')
  counts = {}
  for r in results:
    key = f"{r['status']}/{r['reason']}"
    counts[key] = counts.get(key, 0) + 1
  valid = [r for r in results if r['delta_g'] is not None]
  probe_native = sum(r['probe_steps'] for r in computed)
  continuation_native = sum(r['continuation_steps'] for r in computed)
  unavailable = sum(r['status'] == 'UNAVAILABLE_BUDGET' for r in computed)
  cap = min(probe_steps, max_steps)
  quota = len(chosen) if budget is None else min(budget, len(chosen))
  cost = {key: sum(r['cost'][key] for r in computed) for key in
          ('native_steps', 'policy_calls', 'seconds', 'save_seconds', 'restore_seconds')}
  # A finished branch needs no recovery: its snapshot goes once the output is
  # complete. Unfinished probes keep theirs (a larger quota resumes them).
  finished = [work / f"{r['cursor']}.pkl" for r in computed if r['status'] != 'UNAVAILABLE_BUDGET']
  finished = [p for p in finished if p.exists()]
  meta = dict(
      policy=str(policy), corpus=str(corpus), collection=str(collection),
      gamma=[CF.GAMMA.numerator, CF.GAMMA.denominator], clock='S3_MARKET_MINUTE',
      coupling_floor=reference.floor, reference_rows=reference.n,
      legal_flat_decisions=len(reference.forks()),
      forks=dict(requested_waits=forks, seed=seed, chosen=len(results), requested_cursors=chosen,
                 reference_adds=int(sum(r['reference_action'] == ADD for r in results)),
                 undecided=len(survivors)),
      couple_from=reference.couple_from,
      budget=None if budget is None else dict(forks=budget, undecided=len(survivors),
          simulated=len(sample), unavailable=unavailable, reason='PROBE_UNFINISHED_BUDGET', **strata),
      probe=dict(limit=cap, candidates=len(chosen), native_steps=probe_native,
                 valid_pairs=sum(r['completed_in'] == 'probe' for r in computed)),
      continuation=dict(branches=len(sample), native_steps=continuation_native,
                        valid_pairs=sum(r['completed_in'] == 'continuation' for r in computed)),
      branch_step_bound=len(chosen) * cap + quota * max(0, max_steps - cap),
      end_status={s: sum(r['end_status'] == s for r in results)
                  for s in sorted({r['end_status'] for r in results} - {None})},
      max_steps=max_steps, counts=counts, valid_pairs=len(valid),
      delta_g=dict(mean=float(np.mean([r['delta_g'] for r in valid])) if valid else None,
                   std=float(np.std([r['delta_g'] for r in valid])) if valid else None),
      branch_steps=dict(total=int(sum(r['steps'] for r in computed)),
                        max=int(max((r['steps'] for r in computed), default=0))),
      replay=dict(steps=[reference_metrics['native_steps']] if chosen else [],
                  z_differs=[reference_metrics['z_differs']] if chosen else [],
                  seconds=[reference_metrics['seconds']] if chosen else []),
      compute=dict(reference=reference_metrics, branches=cost,
                   native_steps=reference_metrics.get('native_steps', 0) + cost['native_steps'],
                   policy_calls=reference_metrics.get('policy_calls', 0) + cost['policy_calls'],
                   executed_branches=executed,
                   executed_native_steps=executed['native_steps'] + (
                       reference_metrics.get('native_steps', 0) if not reference_metrics.get('reused') else 0),
                   executed_policy_calls=executed['policy_calls'] + (
                       reference_metrics.get('policy_calls', 0) if not reference_metrics.get('reused') else 0),
                   peak_rss_bytes=memory.peak, memory_method='process-tree sum RSS sampled every 0.2 s',
                   note='Native/call totals include recovered work; elapsed/RSS describe this invocation'),
      seconds=round(time.monotonic() - began, 3),
      inference={k: reference_metrics[k] for k in ('platform', 'compute_dtype')} if chosen else None,
      base_cf=str(base_cf) if base_cf else None,
      snapshots=dict(removed=len(finished), removed_bytes=sum(p.stat().st_size for p in finished),
                     kept=unavailable),
      labels=dict(q_real='STUDENT_EXECUTED_NATIVE', q_cf='VERIFIED_CF_NATIVE',
                  policy_bound='targets are Q^{pi_k} of this policy only'))
  _json_write(out / 'meta.json', meta)
  for p in finished:
    p.unlink()
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
    if r['status'] == 'VALID_COMPLETE' and r['delta_g'] is not None:
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
  `remedial` and `correct` list the cursors of decision errors and of
  correct decisions (S5 feedback, valid pairs only): their rows count
  `remedial_boost` and `correct_boost` times more; every other row is
  sampled as before. `curriculum()` counts what the windows showed.
  """

  def __init__(self, collection, targets, split='train', holdout=2048, burnin=0,
               cf_boost=16., seed=0, carry=None, remedial=None, remedial_boost=1.,
               correct=None, correct_boost=1.):
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
    self.boosts = dict(remedial=float(remedial_boost), correct=float(correct_boost))
    if not all(b > 0 for b in self.boosts.values()):
      raise ValueError(f'Remedial and correct boosts {self.boosts} are positive')
    self.marked = {}
    for kind, cursors in (('remedial', remedial), ('correct', correct)):
      self.marked[kind] = np.zeros(n, bool)
      if cursors is None:
        continue
      at = np.searchsorted(self.cursor, cursors)
      if (at >= n).any() or not np.array_equal(self.cursor[np.minimum(at, n - 1)], cursors):
        raise ValueError(f'{kind.capitalize()} cursors outside the collection')
      if not (self.data['q_cf_valid'][at] > 0).all():
        raise ValueError(f'A {kind} decision needs a valid ADD/WAIT pair at its row')
      self.marked[kind][at] = True
    if (self.marked['remedial'] & self.marked['correct']).any():
      raise ValueError('A decision is both an error and correct')
    self.boost = np.where(self.marked['remedial'], self.boosts['remedial'],
                          np.where(self.marked['correct'], self.boosts['correct'], 1.))
    self.seen = np.zeros(n, bool)
    self.windows = 0
    self.focus_windows = dict.fromkeys(self.marked, 0)

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
      for kind, marked in self.marked.items():
        self.focus_windows[kind] += int(marked[shown].any())

  def sample(self, batch, length):
    starts = self._pick(batch, length)
    self._count(starts, length)
    return self._stack(starts, length)

  def curriculum(self):
    """Q rows the sampled windows showed after the burn-in: distinct rows
    and the exposure of each feedback set (windows that hold one of its
    rows). Repeats of a row are optimizer exposure, never new market samples
    (EDU-R10)."""
    d, seen = self.data, self.seen
    fork = d['q_cf_valid'] > 0
    out = dict(windows=self.windows, unique_q_rows=int((seen & (d['q_real_valid'] > 0)).sum()),
               unique_pairs=int((seen & fork).sum()))
    for kind, marked in self.marked.items():
      out.update({f'{kind}_windows': self.focus_windows[kind], f'{kind}_boost': self.boosts[kind],
                  f'{kind}_rows': int(marked.sum()), f'{kind}_rows_seen': int((marked & seen).sum())})
    return out

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
  `remedial` and `correct` ({contract: cursors of decision errors and of
  correct decisions}, S5 feedback of the same pi_k) boost those rows by
  `remedial_boost` and `correct_boost` inside their contract; the contract
  mix stays as configured."""

  def __init__(self, folders, split='train', holdout=2048, burnin=0, cf_boost=16.,
               seed=0, mix='uniform_contract', carry_root=None, remedial=None,
               remedial_boost=1., correct=None, correct_boost=1.):
    from .corpus import MarketCarry
    self.folders = [Path(f) for f in folders]
    metas = [json.loads((f / 'meta.json').read_text()) for f in self.folders]
    policies = sorted({str(Path(m['policy']).resolve()) for m in metas})
    if len(policies) != 1:
      raise ValueError(f'Q targets of different policies cannot be mixed: {policies}')
    self.policy, self.contracts = policies[0], [_contract(m) for m in metas]
    if len(set(self.contracts)) != len(self.contracts):
      raise ValueError(f'One contract appears twice: {self.contracts}')
    for kind, cursors in (('remedial', remedial), ('correct', correct)):
      unknown = set(cursors or ()) - set(self.contracts)
      if unknown:
        raise ValueError(f'{kind.capitalize()} cursors of contracts without Q targets: '
                         f'{sorted(unknown)}')
    self.parts = [QWindows(m['collection'], f / 'targets.npz', split, holdout, burnin, cf_boost,
                           carry=carry_root and MarketCarry(Path(carry_root) / c),
                           remedial=(remedial or {}).get(c), remedial_boost=remedial_boost,
                           correct=(correct or {}).get(c), correct_boost=correct_boost)
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
             if not k.endswith('_boost')}
    for kind, boost in self.parts[0].boosts.items():
      total[f'{kind}_boost'] = boost
      total[f'{kind}_share'] = total[f'{kind}_windows'] / max(total['windows'], 1)
    return dict(total, contracts=parts, exposure_note=EXPOSURE_NOTE)

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


EXPOSURE_NOTE = ('boosted feedback rows repeat TRAIN rows: windows and repeats are optimizer '
                 'exposure, unique rows count distinct market cursors (EDU-R10)')


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
