"""FG809 P0-P4 corpus as Thinking Student teaching material.

Natural replay windows feed L1. The TRAIN teacher sidecar adds, row for row,
the Native "ADD now, then hold" facts that L_outcome learns from (A6 L2).
Sidecar rows align with the replay NPZ by stepid; a mismatch, or a label on a
state where ADD was not legal, is a data error and raises. Labels are
teaching data only and never enter the policy input.
"""
import gzip
import pickle
from pathlib import Path

import numpy as np

try:
  from orjson import loads
except ImportError:
  from json import loads

from . import labels as L
from .config import ACCOUNT_NATIVE_DIM, observation_space

POLICY_OBS = tuple(observation_space())
REPLAY_KEYS = (*(k for k in POLICY_OBS if k != 'account_native'), 'action')


def native_account(position_state):
  """account_native where the replay row pins it exactly: a flat account is
  [1, direction, 0, ...]. Held rows stay invalid until S2 replays the Native
  account along the recorded actions."""
  flat = (position_state[:, 1] == 0) & (position_state[:, 2] == 0)
  native = np.zeros((len(position_state), ACCOUNT_NATIVE_DIM), np.float32)
  native[flat, 0] = 1.
  native[flat, 1] = position_state[flat, 0]
  return native, flat.astype(np.float32)


def read_facts(path, stepids):
  """Compact teacher L1 facts and the Native legal_add flag per replay row."""
  facts, legal = [], []
  with gzip.open(path, 'rb') as f:
    for i, line in enumerate(f):
      row = loads(line)
      if i >= len(stepids) or row['stepid'] != stepids[i]:
        raise ValueError(f'{path}: row {i} does not align with the replay stepids')
      l1 = (row.get('teacher') or {}).get('L1')
      facts.append({k: l1.get(k) for k in L.L1_FIELDS} if l1 else None)
      legal.append(bool(row['legal_add']))
  if len(facts) != len(stepids):
    raise ValueError(f'{path}: {len(facts)} rows for {len(stepids)} replay steps')
  return facts, np.array(legal)


def policy_facts(corpus, policy, stepids, cache=None):
  """`read_facts` behind an optional pickle cache; the cache is rebuilt when
  it was made from other replay rows or other fact fields."""
  source = Path(corpus) / 'teacher_sidecar' / policy / 'steps.jsonl.gz'
  file = Path(cache) / f'{policy}.l1_facts.pkl' if cache else None
  if file and file.exists():
    with file.open('rb') as f:
      data = pickle.load(f)
    if data['stepids'] == stepids and data['fields'] == L.L1_FIELDS:
      return data['facts'], data['legal']
    print(f'Rebuilding stale label cache {file}')
  facts, legal = read_facts(source, stepids)
  if file:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix('.tmp')
    with tmp.open('wb') as f:
      pickle.dump(dict(stepids=stepids, fields=L.L1_FIELDS, facts=facts,
                       legal=legal), f)
    tmp.rename(file)
  return facts, legal


class CorpusWindows:
  """T-step windows inside one replay chunk, with Outcome labels.

  All policies replay the same market minutes with the same chunk layout,
  so split='train' (all but the last `holdout` chunks of each policy) and
  split='eval' (those chunks) separate one late market period. With labels,
  chunks are sampled in proportion to their labelled rows.
  """

  def __init__(self, corpus, policies=None, seed=0, labels=True, cache=None,
               split='all', holdout=2):
    if split not in ('all', 'train', 'eval'):
      raise ValueError(f'Unknown corpus split {split}')
    root = Path(corpus) / 'policies'
    names = policies or sorted(p.name for p in root.iterdir() if p.is_dir())
    self.chunks, self.counts, self.cache = [], [], {}
    self.labels = {}
    for name in names:
      files = sorted((root / name / 'replay').glob('*.npz'))
      if not files:
        raise FileNotFoundError(f'No replay chunks under {root / name}')
      ids, mask, pos, bounds = [], [], [], []
      for file in files:
        with np.load(file) as d:
          ids += [bytes(row).hex() for row in d['stepid']]
          mask.append(d['action_mask'])
          pos.append(d['position_state'])
        bounds.append((len(ids) - len(mask[-1]), len(ids)))
      if labels:
        facts, legal = policy_facts(corpus, name, ids, cache)
        self.labels[name] = label_check(
            name, L.label_arrays(facts), legal, np.concatenate(mask),
            np.concatenate(pos))
      for i, (file, (lo, hi)) in enumerate(zip(files, bounds)):
        last = i >= len(files) - holdout
        if split == 'all' or (split == 'eval') == last:
          self.chunks.append((name, file, lo, hi))
          self.counts.append(
              self.labels[name]['outcome_valid'][lo:hi].sum() if labels else hi - lo)
    total = float(sum(self.counts))
    if not total:
      raise ValueError('No labelled rows in the selected corpus chunks')
    self.p = np.array(self.counts, np.float64) / total
    self.rng = np.random.default_rng(seed)

  def _chunk(self, c):
    if c not in self.cache:
      name, file, lo, hi = self.chunks[c]
      with np.load(file) as d:
        data = {k: d[k] for k in REPLAY_KEYS}
      data['account_native'], data['account_valid'] = native_account(
          data['position_state'])
      if name in self.labels:
        data.update({k: v[lo:hi] for k, v in self.labels[name].items()})
      else:
        data.update(L.empty_labels((hi - lo,)))
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
          quality=int(lab['outcome_quality_valid'].sum()))
    return out


def label_check(name, labels, legal, mask, position_state):
  """Outcome is defined only for a legal ADD from a flat account (C1-07)."""
  flat = position_state[:, 1] == 0
  ok = legal & (mask[:, 1] > 0.5) & flat
  bad = np.flatnonzero((labels['outcome_valid'] > 0) & ~ok)
  if len(bad):
    raise ValueError(f'{name}: {len(bad)} outcome labels on states without a '
                     f'legal flat ADD, first row {bad[0]}')
  return labels
