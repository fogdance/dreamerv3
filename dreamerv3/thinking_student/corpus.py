"""FG809 P0-P4 corpus as Thinking Student teaching material.

Natural replay windows feed L1. The TRAIN teacher sidecar adds, row for row,
the Native "ADD now, then hold" facts that L_outcome learns from (A6 L2) and
the strict L2 demonstrations that L_policy_demo learns from (A6 L3). Sidecar
rows align with the replay NPZ by stepid; a mismatch, or a label on a state
where ADD was not legal, is a data error and raises. Labels are teaching
data only and never enter the policy input.

Replay rows record position_state but not the account snapshot the student
reads at inference (account_native). `prepare` rebuilds it exactly, once per
policy, by replaying the recorded actions through the Native env, so training
and inference see the same account input on flat and held rows alike.
"""
import contextlib
import gzip
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
  from orjson import loads
except ImportError:
  from json import loads

from . import labels as L
from .config import observation_space

POLICY_OBS = tuple(observation_space())
REPLAY_KEYS = (*(k for k in POLICY_OBS if k != 'account_native'), 'action')
CACHE_ROOT = Path('/data/logdir/fg_single3/thinking_student/corpus_cache')


def default_cache(corpus):
  return CACHE_ROOT / Path(corpus).name


def policy_names(corpus):
  return sorted(p.name for p in (Path(corpus) / 'policies').iterdir() if p.is_dir())


def read_replay(corpus, policy):
  """Chunk files, stepids, action masks, position states and chunk row bounds."""
  files = sorted((Path(corpus) / 'policies' / policy / 'replay').glob('*.npz'))
  if not files:
    raise FileNotFoundError(f'No replay chunks for {policy} under {corpus}')
  ids, mask, pos, bounds = [], [], [], []
  for file in files:
    with np.load(file) as d:
      ids += [bytes(row).hex() for row in d['stepid']]
      mask.append(d['action_mask'])
      pos.append(d['position_state'])
    bounds.append((len(ids) - len(mask[-1]), len(ids)))
  return files, ids, np.concatenate(mask), np.concatenate(pos), bounds


def corpus_factory(corpus):
  """Native E env on the corpus TRAIN source, as the traces were recorded."""
  from e_data_v1.provider import ETrainAdmission
  from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
  corpus = Path(corpus)
  lock = json.loads((corpus / 'source_lock.json').read_text())
  profile = ETrainAdmission.from_train_pool(
      lock['source_pool']['path'], corpus / 'e_source',
      corpus / 'selected_train_pool.json', contracts=[lock['contract']])
  config = corpus / 'inputs' / lock['contract'] / 'env.yaml'
  return lambda: FixedDirectionEntryEnv(config_path=config, e_profile=profile)


def read_facts(path, stepids):
  """Compact teacher L1 facts, strict L2 demonstrations and the Native
  legal_add flag per replay row."""
  facts, demos, legal = [], [], []
  with gzip.open(path, 'rb') as f:
    for i, line in enumerate(f):
      row = loads(line)
      if i >= len(stepids) or row['stepid'] != stepids[i]:
        raise ValueError(f'{path}: row {i} does not align with the replay stepids')
      teacher = row.get('teacher') or {}
      l1, l2 = teacher.get('L1'), teacher.get('L2')
      facts.append({k: l1.get(k) for k in L.L1_FIELDS} if l1 else None)
      demos.append({k: l2.get(k) for k in L.L2_FIELDS} if l2 else None)
      legal.append(bool(row['legal_add']))
  if len(facts) != len(stepids):
    raise ValueError(f'{path}: {len(facts)} rows for {len(stepids)} replay steps')
  return facts, demos, np.array(legal)


def policy_facts(corpus, policy, stepids, cache=None):
  """`read_facts` behind an optional pickle cache; the cache is rebuilt when
  it was made from other replay rows or other fact fields."""
  source = Path(corpus) / 'teacher_sidecar' / policy / 'steps.jsonl.gz'
  file = Path(cache) / f'{policy}.l1_facts.pkl' if cache else None
  fields = (L.L1_FIELDS, L.L2_FIELDS)
  if file and file.exists():
    with file.open('rb') as f:
      data = pickle.load(f)
    if data['stepids'] == stepids and data['fields'] == fields:
      return data['facts'], data['demos'], data['legal']
    print(f'Rebuilding stale label cache {file}')
  facts, demos, legal = read_facts(source, stepids)
  if file:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix('.tmp')
    with tmp.open('wb') as f:
      pickle.dump(dict(stepids=stepids, fields=fields, facts=facts,
                       demos=demos, legal=legal), f)
    tmp.rename(file)
  return facts, demos, legal


def policy_accounts(corpus, policy, stepids, cache, factory=None):
  """Exact account_native of every replay row of one policy, from one Native
  replay of its recorded actions; the cache is rebuilt when it was made from
  other replay rows."""
  file = Path(cache) / f'{policy}.account_native.npz'
  if file.exists():
    with np.load(file) as d:
      if d['stepids'].tolist() == stepids:
        return d['account_native']
    print(f'Rebuilding stale account cache {file}')
  from gym_trading_env.research.ts_v1_account import replay_account_native
  ids, native = replay_account_native(
      Path(corpus) / 'policies' / policy, factory or corpus_factory(corpus))
  if [bytes(row).hex() for row in ids] != stepids:
    raise ValueError(f'{policy}: Native replay rows do not align with the replay chunks')
  file.parent.mkdir(parents=True, exist_ok=True)
  tmp = file.with_suffix('.tmp.npz')
  np.savez(tmp, stepids=np.array(stepids), account_native=native)
  tmp.rename(file)
  return native


def account_check(name, native, position_state):
  """The replayed account must agree with the position facts that the same
  replay rows recorded: flat flag, direction, held lots, TP distance, age."""
  held = position_state[:, 1] != 0
  checks = dict(
      flat=native[:, 0] == ~held,
      direction=native[:, 1] == position_state[:, 0],
      held_ratio=np.isclose(native[:, 2], position_state[:, 1]),
      tp_distance=~held | np.isclose(native[:, 3], position_state[:, 4], rtol=1e-5),
      held_minutes=~held | np.isclose(native[:, 6], position_state[:, 5], rtol=1e-5))
  for key, ok in checks.items():
    if not ok.all():
      raise ValueError(f'{name}: replayed account {key} disagrees with '
                       f'position_state at row {np.flatnonzero(~ok)[0]}')
  return native


def _prepare_policy(job):
  corpus, cache, name, labels = job
  began = time.monotonic()
  _, ids, _, pos, _ = read_replay(corpus, name)
  if labels:
    policy_facts(corpus, name, ids, cache)  # facts and demonstrations
  native = account_check(name, policy_accounts(corpus, name, ids, cache), pos)
  return dict(policy=name, rows=len(ids), held=int((native[:, 0] == 0).sum()),
              seconds=round(time.monotonic() - began, 1))


@contextlib.contextmanager
def _cpu_workers():
  """Environment inherited by spawned workers: CPU only, one native thread."""
  keys = dict(JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
              MKL_NUM_THREADS='1')
  saved = {k: os.environ.get(k) for k in keys}
  os.environ.update(keys)
  try:
    yield
  finally:
    for k, v in saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v


def prepare(corpus, cache, policies=None, labels=True, workers=None):
  """Build the label and exact-account caches, one spawned CPU process per
  policy (a policy's account replay is sequential; policies are independent)."""
  jobs = [(str(corpus), str(cache), name, labels)
          for name in policies or policy_names(corpus)]
  workers = min(len(jobs), workers or os.cpu_count())
  with _cpu_workers(), ProcessPoolExecutor(
      workers, mp_context=multiprocessing.get_context('spawn')) as pool:
    return list(pool.map(_prepare_policy, jobs))


class CorpusWindows:
  """T-step windows inside one replay chunk, with exact Native accounts,
  Outcome labels and strict L2 demonstrations.

  All policies replay the same market minutes with the same chunk layout,
  so split='train' (all but the last `holdout` chunks of each policy) and
  split='eval' (those chunks) separate one late market period. With labels,
  chunks are sampled in proportion to their Outcome labels (weight='outcome')
  or demonstrations (weight='demo'). Demonstrations carry class weights
  (`L.class_weights` over the selected chunks unless `demo_weights` is
  given, e.g. the train split's for an eval split). Missing caches are
  built first with `prepare`.
  """

  def __init__(self, corpus, cache=None, policies=None, seed=0, labels=True,
               split='all', holdout=2, workers=None, weight='outcome',
               demo_balance=1.0, demo_weights=None):
    if split not in ('all', 'train', 'eval'):
      raise ValueError(f'Unknown corpus split {split}')
    if weight not in ('outcome', 'demo'):
      raise ValueError(f'Unknown corpus sampling weight {weight}')
    cache = Path(cache or default_cache(corpus))
    names = list(policies or policy_names(corpus))
    missing = [n for n in names if not (cache / f'{n}.account_native.npz').exists()
               or (labels and not (cache / f'{n}.l1_facts.pkl').exists())]
    if missing:
      print('Prepared corpus cache', cache, prepare(corpus, cache, missing, labels, workers))
    self.chunks, self.counts, self.cache = [], [], {}
    self.labels, self.accounts = {}, {}
    for name in names:
      files, ids, mask, pos, bounds = read_replay(corpus, name)
      self.accounts[name] = account_check(
          name, policy_accounts(corpus, name, ids, cache), pos)
      if labels:
        facts, demos, legal = policy_facts(corpus, name, ids, cache)
        action, valid = L.demo_arrays(demos)
        self.labels[name] = label_check(
            name, {**L.label_arrays(facts), 'demo_action': action,
                   'demo_valid': valid}, legal, mask, pos)
      for i, (file, (lo, hi)) in enumerate(zip(files, bounds)):
        last = i >= len(files) - holdout
        if split == 'all' or (split == 'eval') == last:
          self.chunks.append((name, file, lo, hi))
          key = 'outcome_valid' if weight == 'outcome' else 'demo_valid'
          self.counts.append(self.labels[name][key][lo:hi].sum() if labels else hi - lo)
    total = float(sum(self.counts))
    if not total:
      raise ValueError('No labelled rows in the selected corpus chunks')
    self.p = np.array(self.counts, np.float64) / total
    self.rng = np.random.default_rng(seed)
    self.demo_counts, self.demo_weights = None, None
    if labels:
      action, valid = (np.concatenate([
          self.labels[name][k][lo:hi] for name, _, lo, hi in self.chunks])
          for k in ('demo_action', 'demo_valid'))
      self.demo_counts = dict(wait=int((valid & (action == 0)).sum()),
                              add=int((valid & (action == 1)).sum()))
      self.demo_weights = tuple(demo_weights) if demo_weights else (
          L.class_weights(action, valid, demo_balance) if weight == 'demo' else (1., 1.))

  def _chunk(self, c):
    if c not in self.cache:
      name, file, lo, hi = self.chunks[c]
      with np.load(file) as d:
        data = {k: d[k] for k in REPLAY_KEYS}
      data['account_native'] = self.accounts[name][lo:hi]
      data['account_valid'] = np.ones(hi - lo, np.float32)
      data.update(L.empty_labels((hi - lo,)))
      if name in self.labels:
        lab = {k: v[lo:hi] for k, v in self.labels[name].items()}
        valid, action = lab.pop('demo_valid'), lab['demo_action']
        lab['demo_weight'] = np.where(
            valid, np.where(action == 1, self.demo_weights[1], self.demo_weights[0]),
            0.).astype(np.float32)
        data.update(lab)
      self.cache[c] = data
    return self.cache[c]

  def _stack(self, windows, length):
    rows = [{k: v[s:s + length] for k, v in self._chunk(c).items()}
            for c, s in windows]
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    data['is_first'][:, 0] = True
    return data

  def sample(self, batch, length):
    windows = []
    for _ in range(batch):
      c = int(self.rng.choice(len(self.chunks), p=self.p))
      n = self.chunks[c][3] - self.chunks[c][2]
      windows.append((c, int(self.rng.integers(0, n - length + 1))))
    return self._stack(windows, length)

  def batches(self, batch, length):
    """Every non-overlapping window of the sampled chunks, in full batches."""
    windows = [(c, s) for c in range(len(self.chunks)) if self.p[c] > 0
               for s in range(0, self.chunks[c][3] - self.chunks[c][2] - length + 1, length)]
    for i in range(0, len(windows) - batch + 1, batch):
      yield self._stack(windows[i:i + batch], length)

  def summary(self):
    out = {}
    for name, lab in self.labels.items():
      allowed = lab['outcome_allowed']
      single = allowed.sum(-1) == 1
      out[name] = dict(
          rows=len(allowed), labels=int(lab['outcome_valid'].sum()),
          tp=int((single & (allowed[:, :L.NBINS].sum(-1) == 1)).sum()),
          sl=int((single & (allowed[:, L.NBINS:2 * L.NBINS].sum(-1) == 1)).sum()),
          survive=int((single & (allowed[:, L.SURVIVE] == 1)).sum()),
          censored=int(((allowed.sum(-1) > 1)).sum()),
          quality=int(lab['outcome_quality_valid'].sum()),
          demo_wait=int((lab['demo_valid'] & (lab['demo_action'] == 0)).sum()),
          demo_add=int((lab['demo_valid'] & (lab['demo_action'] == 1)).sum()))
    return out


def label_check(name, labels, legal, mask, position_state):
  """Outcome and the WAIT/ADD choice exist only where a flat account may
  legally ADD (C1-07); a label anywhere else is a data error."""
  flat = position_state[:, 1] == 0
  ok = legal & (mask[:, 1] > 0.5) & flat
  for key in ('outcome_valid', 'demo_valid'):
    bad = np.flatnonzero((labels[key] > 0) & ~ok)
    if len(bad):
      raise ValueError(f'{name}: {len(bad)} {key} labels on states without a '
                       f'legal flat ADD, first row {bad[0]}')
  return labels
