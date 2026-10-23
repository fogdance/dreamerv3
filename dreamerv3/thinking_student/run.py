"""Thinking Student V1 runner: staged training on TS teaching material, frozen
policy snapshots and autonomous Native experience.

  python -m dreamerv3.thinking_student.run train-market  --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run train-outcome --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run train-policy  --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run freeze  --run DIR --name pi0 --corpus DIR
  python -m dreamerv3.thinking_student.run collect --policy DIR --corpus DIR --out DIR
  python -m dreamerv3.thinking_student.run check   --out DIR [--corpus DIR]
  python -m dreamerv3.thinking_student.run rollout --run DIR --corpus DIR
  python -m dreamerv3.thinking_student.run cf      --policy DIR --corpus DIR [--forks N]
  python -m dreamerv3.thinking_student.run train-value --run DIR --init-from POLICY --cf DIR
  python -m dreamerv3.thinking_student.run freeze  --run DIR --name pi1 --beta-q 1 --kappa-q auto --cf DIR

--corpus is one contract folder of gym ts_v1_corpus (teaching material and
the TS env inputs: Quality TP reward, no CHUNK_END).

A run directory holds `ckpt/<step>/agent.pkl` and a `latest` pointer. Loading
never checks code or data hashes (C2); shape mismatches fail loudly instead
of silently re-initializing. A frozen policy `policies/<name>/` holds the
complete parameters and `policy.json`, the record of everything that decides
its actions (A6 2.4); collection and exams load it and never train it.
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from . import labels as L
from .config import observation_space
from .corpus import LABEL_ACCOUNT, CorpusWindows, TSCorpus
from .student import make_agent

POLICY_OBS = tuple(observation_space())
DECISION_OBS = tuple(k for k in POLICY_OBS
                     if k not in ('reward', 'is_first', 'is_last', 'is_terminal'))


# ---- checkpoint ----

def save(agent, run, tag=None):
  run = Path(run)
  data = agent.save()
  name = tag or f"{data['counters']['updates']:012d}"
  folder = run / 'ckpt' / name
  folder.mkdir(parents=True, exist_ok=True)
  tmp = folder / 'agent.pkl.tmp'
  with tmp.open('wb') as f:
    pickle.dump(data, f)
  tmp.rename(folder / 'agent.pkl')
  (run / 'ckpt' / 'latest').write_text(name)
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


def build(run, preset, device, init_f1=None, init_from=None, migrate=None, **kwargs):
  """Resume the run strictly; a fresh run may start from F1 market weights
  or from another Thinking Student checkpoint (`init_from`, listed changes,
  optionally through a named layout `migrate`)."""
  if init_f1 and init_from:
    raise ValueError('Choose one of init_f1 and init_from')
  agent = make_agent(Path(run) / 'logdir', preset=preset, device=device, **kwargs)
  folder = resume(agent, run)
  if folder is not None:
    print(f'Resumed {folder}')
  elif init_from:
    from .transfer import inherit_student
    lists = inherit_student(agent, init_from, migrate)
    Path(run).mkdir(parents=True, exist_ok=True)
    (Path(run) / 'inherit.json').write_text(json.dumps(lists, indent=1))
  elif init_f1:
    from .transfer import inherit_f1
    inherit_f1(agent, init_f1)
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


def freeze(agent, run, name, preset, corpus=None, teaching=None):
  """Write `run/policies/<name>/`: the complete parameters (every module
  that can change an action) and policy.json with the action selection,
  decision coefficients, RSSM initialization, Reward and data sources."""
  folder = Path(run) / 'policies' / name
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
          'training windows start from the zero carry with a loss burn-in; '
          'stochastic latents use the policy call counter seeds'),
      outcome=dict(horizon_minutes=L.HORIZON, bins=list(L.OUTCOME_BINS),
                   classes=[*(f'{e}_bin{k}' for e in L.EVENTS for k in range(L.NBINS)),
                            'OPEN_AT_H'],
                   answers='p_tp_h, p_sl_h, p_other_h, p_open_h: first ending within H'),
      untrained=dict(reason_pref='diagnostic only; L_reason trains it from S4'),
      preset=preset, ts=_flat(dict(agent.config.ts), ''),
      # z near 0 can change sign between compute dtypes; collection, CF and
      # exams load the policy on the device and dtype recorded here.
      inference=dict(platform=agent.config.jax.platform,
                     compute_dtype=agent.config.jax.compute_dtype),
      teaching=teaching or {}, repos=_repos())
  if corpus:
    source = json.loads((Path(corpus) / 'source.json').read_text())
    meta['corpus'] = str(corpus)
    meta['reward_version'] = source['inputs']['reward_schema']
    meta['teacher'] = {k: source[k] for k in (
        'contract', 'fold', 'mode', 'funds_floor', 'gamma', 'clock', 'created')}
  tmp = folder / 'agent.pkl.tmp'
  tmp.write_bytes(blob)
  tmp.rename(folder / 'agent.pkl')
  (folder / 'policy.json').write_text(json.dumps(meta, indent=1))
  return folder


def load_policy(folder, device=None, logdir=None, log=print):
  """Build the student exactly as the snapshot recorded it (by default on
  its recorded inference device and dtype) and load it. The snapshot's
  non-optimizer parameters fix its actions and must all match; optimizer
  state of a training stage added after the freeze never acts and starts
  fresh (listed)."""
  folder = Path(folder)
  meta = json.loads((folder / 'policy.json').read_text())
  updates = {f'agent.ts.{k}': tuple(v) if isinstance(v, list) else v
             for k, v in meta['ts'].items()}
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
               'p_sl_h', 'p_other_h', 'p_open_h', 'quality')
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
  then state.pkl, so a stopped collection resumes at its last full chunk."""
  out = Path(out)
  for sub in ('steps', 'replay'):
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


FACT_KINDS = {'TAKE_PROFIT': 'TP5', 'STOP_LOSS': 'SL30'}


def check_collection(out, corpus=None):
  """Minimal verification of an autonomous collection (C3 S2): each recorded
  proposal follows the formal rule, the formal pi executed its own proposal
  (forced counterfactual actions excepted), no executed ADD was masked, and
  each closed Native trade reached TP by >= 5 points or lost >= 30 points at
  SL. With the corpus of the collection's contract, each student ADD from a
  funded flat account is compared with the Native fact of the same cursor."""
  out = Path(out)
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
    if (t['reason'] == 'TAKE_PROFIT' and move < 5) or (
        t['reason'] == 'STOP_LOSS' and move > -30):
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


# ---- staged training on natural corpus sequences ----

LOG_KEYS = {
    'L1': ('opt_l1/loss', 'loss/dyn', 'loss/market_seq_e_shared',
           'loss/account_can_add', 'brain/p_add', 'brain/delta_q_abs'),
    'L2': ('opt_l2/loss', 'loss/outcome_event', 'loss/outcome_risk',
           'loss/outcome_mae', 'loss/outcome_quality', 'outcome/labels',
           'outcome/p_tp_on_tp', 'outcome/p_tp_on_sl', 'outcome/p_other_h'),
    'L3': ('opt_l3/loss', 'loss/policy_demo', 'policy/n_add',
           'policy/p_add_on_add', 'policy/p_add_on_wait', 'policy/add_rate',
           'brain/alpha'),
    'L4': ('opt_l4/loss', 'loss/q_real', 'loss/q_cf', 'loss/q_pair', 'q/n_cf',
           'q/delta_q_on_cf', 'q/delta_g', 'q/sign_agree', 'brain/delta_q_abs'),
}


def _device(agent, data, counter):
  from embodied.jax import internal
  data = internal.device_put(data, agent.train_sharded)
  data['seed'] = agent._seeds(counter, agent.train_mirrored)
  return data


def train_stage(agent, windows, updates, log_every=10, on_update=None):
  """`updates` optimizer steps of the configured A6 stage on corpus windows."""
  stage, mets = agent.config.ts.stage, {}
  for i in range(updates):
    data = windows.sample(agent.config.batch_size, agent.config.batch_length)
    data = _device(agent, data, int(agent.n_batches))
    agent.n_batches.increment()
    _, _, mets = agent.train(agent.init_train(agent.config.batch_size), data)
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
        'loss/outcome_quality': 'outcome/n_quality',
        'outcome/p_tp_on_tp': 'outcome/n_tp', 'outcome/p_tp_on_sl': 'outcome/n_sl'}),
    'L3': (('loss/', 'policy/'), 'policy/n_demo', {
        'policy/p_add_on_add': 'policy/n_add', 'policy/acc_add': 'policy/n_add',
        'policy/p_add_on_wait': 'policy/n_wait', 'policy/acc_wait': 'policy/n_wait',
        'policy/add_rate': 'policy/n_legal'}),
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
      w = np.asarray(warm(data['is_first'], ts.demo_burnin)) * data['demo_weight']
      n = float((w > 0).sum())
      pool('constant/policy_demo', np.log(2.) * w.sum() / max(n, 1.), n)
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
    _, mets = agent.report(agent.init_report(B), _device(agent, data, i))
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


def _train_with_eval(agent, args, train, held, report, constant=None):
  def checkpoint(done=None):
    if done is None or (args.eval_every and done % args.eval_every == 0
                        and done != args.updates):
      report['eval'].append(evaluate(agent, held, constant))
      print('eval', report['eval'][-1])

  stage = agent.config.ts.stage
  checkpoint()
  report['last_train'] = {k: float(v) for k, v in train_stage(
      agent, train, args.updates, args.log_every, checkpoint).items()
      if k.startswith(LOG_KEYS[stage])}
  checkpoint()
  return save(agent, args.run)


def value_anchor(args):
  """beta_Q and kappa_Q of a new snapshot (A5); kappa 'auto' is std(Delta G)
  over the valid train pairs of the --cf targets."""
  record = dict(beta_q=float(args.beta_q))
  if args.kappa_q == 'auto':
    from .value import QWindows
    meta = json.loads((Path(args.cf) / 'meta.json').read_text())
    kappa, pairs = QWindows(meta['collection'], Path(args.cf) / 'targets.npz', 'train',
                            holdout=args.holdout).kappa()
    record.update(kappa_q=kappa, kappa_source=dict(cf=str(args.cf), split='train',
                                                    holdout=args.holdout, pairs=pairs))
  else:
    record['kappa_q'] = float(args.kappa_q)
  if not record['kappa_q'] > 0:
    raise ValueError(f'kappa_Q must be positive, got {record["kappa_q"]}')
  return record


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('command', choices=[
      'train-market', 'train-outcome', 'train-policy', 'freeze', 'collect',
      'check', 'rollout', 'cf', 'train-value'])
  parser.add_argument('--run', default=None)
  parser.add_argument('--preset', default='size12m')
  parser.add_argument('--device', default=None,
                      help="cpu or cuda; default cuda (collect: the snapshot's device)")
  parser.add_argument('--init-f1', default=None)
  parser.add_argument('--init-from', default=None,
                      help='earlier Thinking Student run or checkpoint (listed inheritance)')
  parser.add_argument('--migrate', default=None, choices=['r2r3'],
                      help='parameter layout change of --init-from (transfer.MIGRATIONS)')
  parser.add_argument('--corpus', default=None, help='TS corpus folder of one contract')
  parser.add_argument('--holdout', type=int, default=2048,
                      help='last market cursors held out of training')
  parser.add_argument('--updates', type=int, default=10)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--batch-length', type=int, default=64)
  parser.add_argument('--log-every', type=int, default=10)
  parser.add_argument('--eval-every', type=int, default=0)
  parser.add_argument('--demo-balance', type=float, default=1.0,
                      help='L3 class weighting: 1 equal class weight, 0 natural')
  parser.add_argument('--name', default='pi0', help='frozen policy name')
  parser.add_argument('--policy', default=None, help='frozen policy folder')
  parser.add_argument('--out', default=None, help='collection folder')
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
  parser.add_argument('--cf', default=None, help='cf output folder (train-value, kappa)')
  parser.add_argument('--cf-boost', type=float, default=16.,
                      help='train-value: sampling weight of a fork row with a valid branch')
  parser.add_argument('--beta-q', type=float, default=None,
                      help='freeze: A5 value-anchor weight beta_Q')
  parser.add_argument('--kappa-q', default=None,
                      help="freeze: kappa_Q, a number or 'auto' = std(Delta G) of --cf train pairs")
  args = parser.parse_args(argv)
  if args.command == 'collect':
    agent, meta = load_policy(args.policy, args.device)
    out = args.out or Path(args.policy).parents[1] / 'collect' / f"{meta['policy']}_{args.mode}"
    print('Collecting into', out)
    collect(agent, corpus_env(args.corpus), out, args.steps, args.mode, args.chunk,
            meta['policy'])
    return
  if args.command == 'check':
    report = check_collection(args.out, args.corpus)
    (Path(args.out) / 'check.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return
  if args.command == 'cf':
    from .value import run_cf
    policy = Path(args.policy)
    name = json.loads((policy / 'policy.json').read_text())['policy']
    collection = args.collection or policy.parents[1] / 'collect' / f'{name}_policy'
    out = Path(args.out or policy.parents[1] / 'cf' / name)
    run_cf(policy, args.corpus, collection, out, args.forks, args.seed, args.workers,
           args.max_steps)
    return
  if not args.run:
    parser.error(f'{args.command} needs --run')
  stage = {'train-market': 'L1', 'train-outcome': 'L2', 'train-policy': 'L3',
           'train-value': 'L4'}.get(args.command, 'L1')
  updates, anchor = {'agent.ts.stage': stage}, None
  if args.command == 'freeze' and args.beta_q is not None:
    anchor = value_anchor(args)
    updates.update({'agent.ts.decision.beta_q': anchor['beta_q'],
                    'agent.ts.decision.kappa_q': anchor['kappa_q']})
  agent = build(args.run, args.preset, args.device or 'cuda', args.init_f1, args.init_from,
                args.migrate, batch_size=args.batch_size, batch_length=args.batch_length,
                updates=updates)
  ts = agent.config.ts
  if args.command in ('train-market', 'train-outcome', 'train-policy'):
    corpus = TSCorpus(args.corpus)
  if args.command == 'train-market':
    windows = CorpusWindows(corpus, split='train', holdout=args.holdout)
    train_stage(agent, windows, args.updates, args.log_every)
    print('Saved', save(agent, args.run))
  elif args.command == 'train-outcome':
    kw = dict(weight='outcome', burnin=ts.outcome_burnin, holdout=args.holdout,
              accounts=[LABEL_ACCOUNT])
    train = CorpusWindows(corpus, split='train', **kw)
    held = CorpusWindows(corpus, split='eval', **kw)
    constant = constant_answer(train)
    report = {'corpus': args.corpus, 'holdout': args.holdout,
              'train_labels': train.summary(), 'eval_labels': held.summary(),
              'constant_answer': constant.round(5).tolist(), 'eval': []}
    folder = _train_with_eval(agent, args, train, held, report, constant)
    (folder / 'train_outcome.json').write_text(json.dumps(report, indent=1))
    print('Saved', folder)
  elif args.command == 'train-policy':
    kw = dict(weight='demo', burnin=ts.demo_burnin, holdout=args.holdout,
              accounts=[LABEL_ACCOUNT])
    train = CorpusWindows(corpus, split='train', demo_balance=args.demo_balance, **kw)
    held = CorpusWindows(corpus, split='eval', demo_weights=train.demo_weights, **kw)
    report = {'corpus': args.corpus, 'holdout': args.holdout,
              'demo_balance': args.demo_balance, 'demo_weights': train.demo_weights,
              'train_demos': train.demo_counts, 'eval_demos': held.demo_counts,
              'eval': []}
    print('demonstrations', {k: report[k] for k in (
        'demo_weights', 'train_demos', 'eval_demos')})
    folder = _train_with_eval(agent, args, train, held, report)
    (folder / 'train_policy.json').write_text(json.dumps(report, indent=1))
    print('Saved', folder)
  elif args.command == 'train-value':
    from .value import QWindows
    cf = Path(args.cf)
    meta = json.loads((cf / 'meta.json').read_text())
    kw = dict(holdout=args.holdout, burnin=ts.q_burnin)
    train = QWindows(meta['collection'], cf / 'targets.npz', 'train',
                     cf_boost=args.cf_boost, seed=args.seed, **kw)
    held = QWindows(meta['collection'], cf / 'targets.npz', 'eval', **kw)
    constant = train.constant()
    report = {'cf': str(cf), 'policy': meta['policy'], 'holdout': args.holdout,
              'cf_boost': args.cf_boost, 'constant': constant,
              'train_pairs': train.kappa()[1], 'eval_pairs': held.kappa()[1], 'eval': []}
    print('Q targets', {k: report[k] for k in ('constant', 'train_pairs', 'eval_pairs')})
    folder = _train_with_eval(agent, args, train, held, report, constant)
    (folder / 'train_value.json').write_text(json.dumps(report, indent=1))
    print('Saved', folder)
  elif args.command == 'freeze':
    teaching = {}
    for name in ('train_policy', 'train_value'):
      report = latest(args.run) / f'{name}.json'
      if report.exists():
        data = json.loads(report.read_text())
        teaching[name] = {k: v for k, v in data.items() if k != 'eval'}
        teaching[name]['final_eval'] = data['eval'][-1]
    if anchor:
      teaching['value_anchor'] = anchor
    print('Frozen', freeze(agent, args.run, args.name, args.preset, args.corpus, teaching))
  else:
    out = Path(args.run) / 'rollouts'
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'rollout_{int(agent.n_actions):09d}.jsonl').open('w') as f:
      _, rows = rollout(agent, corpus_env(args.corpus), args.steps, args.mode, record=f)
    adds = sum(r['executed'] == 1 for r in rows)
    print(f'rollout steps={len(rows)} adds={adds} '
          f'reward={sum(r["reward"] for r in rows):.3f}')
    if not latest(args.run):
      print('Saved', save(agent, args.run))


if __name__ == '__main__':
  main()
