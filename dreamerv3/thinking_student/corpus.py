"""Thinking Student teaching material (gym ts_v1_corpus) as training windows.

A TS corpus holds, per TRAIN contract and market cursor, the causal market
observation (market/), the Native "ADD now, then hold" facts (facts/) and the
strict TRAIN oracle's WAIT/ADD (strict/) of a flat funded account, plus the
account side of the behavior accounts P0-P4 (behaviors/<P>/, one row per
cursor from the first market minute to the true terminal).

A window is T consecutive rows of one behavior account joined by cursor with
the market rows. Outcome labels and demonstrations sit exactly on the rows
where that account is flat, may ADD, keeps the funds floor and is not at its
terminal: there the cursor's facts and strict choice are this account's own,
because the market alone decides them. Labels are teaching data only and
never enter the policy input.

Every account replays the same market minutes, and a label row reads the
same flat account and market at its cursor in every account. Label windows
therefore come from LABEL_ACCOUNT, the always-WAIT account that is flat at
every cursor: each market cursor's label counts exactly once (C3 EDU-R10),
while L1 reads every account for the held account states.

Formal training (C4 2) reads several TRAIN contracts at once: `discover`
lists the corpora of one root, MultiCorpusWindows mixes their windows.
"""
import json
from pathlib import Path

import numpy as np

from gym_trading_env.research import ts_v1_corpus as C

from . import labels as L

ACCOUNT_KEYS = ('position_state', 'action_mask', 'account_native', 'is_last',
                'is_terminal')
LABEL_ACCOUNT = 'P0'


class TSCorpus:
  """One contract's teaching material, loaded once and shared by windows.
  The market arrays (most of the material) are memory-mapped: windows read
  their rows, and many contracts share the page cache instead of each
  holding a full copy (C4 2.1)."""

  def __init__(self, folder, behaviors=None):
    self.folder = Path(folder)
    self.source = json.loads((self.folder / 'source.json').read_text())
    self.market = C.load_arrays(self.folder / 'market', mmap='r')
    rows = self.source['rows']
    for k in C.MARKET_KEYS:
      _table_check(f'market/{k}', self.market[k], rows)
    facts = C.load_arrays(self.folder / 'facts')
    self.outcome = L.outcome_labels(facts, C.KINDS)
    self.demo_action, self.demo_valid = L.demo_labels(
        C.load_arrays(self.folder / 'strict'))
    floor = self.source['funds_floor']
    self.behaviors, self.labelled = {}, {}
    for name in behaviors or self.source['behaviors']:
      b = C.load_arrays(self.folder / 'behaviors' / name)
      for k in ('cursor', 'equity', 'action', 'reward', 'is_first', *ACCOUNT_KEYS):
        _table_check(f'{name}/{k}', b[k], rows)
      funded = (b['position_state'][:, 1] == 0) & (b['equity'] >= floor) & ~b['is_last']
      legal = b['action_mask'][:, 1] > 0.5
      # The cursor's facts are this account's only if a funded flat account
      # may ADD exactly where the fact says the ADD is eligible.
      bad = np.flatnonzero(funded & (legal != facts['eligible'][b['cursor']]))
      if len(bad):
        raise ValueError(f'{name}: ADD legality differs from the facts at '
                         f'{len(bad)} funded flat rows, first cursor {b["cursor"][bad[0]]}')
      # The observation of a row carries the reward of the previous action.
      b['obs_reward'] = np.concatenate([[0.], b['reward'][:-1]]).astype(np.float32)
      self.behaviors[name] = b
      self.labelled[name] = funded & legal

  @property
  def rows(self):
    return self.source['rows']

  def window(self, name, lo, hi, market=True):
    """Student inputs, actions and labels of rows [lo, hi) of one account;
    `demo_valid` marks the rows whose demonstration counts. market=False
    leaves out the market observation (label counts)."""
    b = self.behaviors[name]
    cursor = b['cursor'][lo:hi]
    on = self.labelled[name][lo:hi]
    data = {k: np.asarray(self.market[k][cursor]) for k in C.MARKET_KEYS} if market else {}
    data.update({k: b[k][lo:hi] for k in ACCOUNT_KEYS})
    data.update(
        reward=b['obs_reward'][lo:hi], is_first=b['is_first'][lo:hi].copy(),
        action=b['action'][lo:hi].astype(np.int32),
        account_valid=np.ones(hi - lo, np.float32))
    for k, v in self.outcome.items():
      data[k] = v[cursor] * on.reshape((-1,) + (1,) * (v.ndim - 1))
    data['demo_valid'] = on & self.demo_valid[cursor]
    data['demo_action'] = np.where(
        data['demo_valid'], self.demo_action[cursor], 0.).astype(np.float32)
    # Behavior accounts are not pi_k: no Q^{pi_k} targets (A6 2.4).
    data.update({k: np.zeros(hi - lo, np.float32) for k in L.Q_SPACE})
    return data


def _table_check(name, array, rows):
  """Row count of a table; finite values of a loaded (not mapped) one."""
  if len(array) != rows:
    raise ValueError(f'{name}: {len(array)} rows for {rows} market cursors')
  if (array.dtype.kind == 'f' and not isinstance(array, np.memmap)
      and not np.isfinite(array).all()):
    raise ValueError(f'{name}: values are not finite')


REQUIRED = ('source.json', 'market', 'facts', 'strict', 'behaviors', 'inputs/env.yaml')


def corpus_sources(folders):
  """source.json of each TRAIN corpus folder. All must be TRAIN material of
  distinct contracts with one Native reward, clock and discount; a folder
  that differs is named and refused (C4 2.1)."""
  sources, shared, seen = [], None, {}
  for folder in map(Path, folders):
    missing = [k for k in REQUIRED if not (folder / k).exists()]
    if missing:
      raise ValueError(f'{folder}: not a TS corpus, missing {missing}')
    source = json.loads((folder / 'source.json').read_text())
    name = source['contract']
    if source['fold'] != 'TRAIN':
      raise ValueError(f'{name}: fold {source["fold"]} is not TRAIN material')
    key = dict(reward_schema=source['inputs']['reward_schema'], clock=source['clock'],
               gamma=source['gamma'])
    if shared is not None and key != shared[1]:
      raise ValueError(f'{name}: {key} differs from {shared[0]} {shared[1]}')
    if name in seen:
      raise ValueError(f'{name} appears twice: {seen[name]} and {folder}')
    shared, seen[name] = shared or (name, key), folder
    sources.append(source)
  return sources


def discover(root, contracts='discover'):
  """The TS corpus folders under `root`: every folder with a source.json
  ('discover'), or the listed contracts. The list comes from the disk, never
  from the code."""
  root = Path(root)
  if contracts == 'discover':
    folders = sorted(p for p in root.iterdir() if (p / 'source.json').exists())
  else:
    folders = [root / name for name in contracts]
  if not folders:
    raise ValueError(f'No TS corpus under {root}')
  corpus_sources(folders)
  return folders


class CorpusWindows:
  """T-step windows of the behavior accounts with their labels.

  Every account replays the same market minutes, so split='train' (windows
  ending before the last `holdout` cursors) and split='eval' (windows inside
  them) separate one late market period; split='all' (or holdout=0) uses every cursor.
  A window starts from the zero market state (is_first) like the policy at
  the first market minute: an encoder warm-up on a slice of one continuous trace,
  not a data break; its first `burnin` rows carry no label loss.
  weight='outcome' / 'demo' samples windows in proportion to their labelled
  rows after the burn-in, weight=None uniformly. `accounts` selects the
  behavior accounts (default: all). Demonstrations carry class weights
  (`L.class_weights` over the split's demonstrations unless `demo_weights`
  is given, e.g. the train split's for an eval split).
  """

  def __init__(self, corpus, split='all', weight=None, burnin=0, holdout=2048,
               seed=0, demo_balance=1.0, demo_weights=None, accounts=None):
    if split not in ('all', 'train', 'eval'):
      raise ValueError(f'Unknown corpus split {split}')
    if weight not in (None, 'outcome', 'demo'):
      raise ValueError(f'Unknown corpus sampling weight {weight}')
    n = corpus.rows
    if not 0 <= holdout < n or (split == 'eval' and not holdout):
      raise ValueError(f'Holdout of {holdout} cursors does not fit {n} rows for split {split}')
    self.corpus, self.weight, self.burnin = corpus, weight, burnin
    self.lo, self.hi = {'all': (0, n), 'train': (0, n - holdout),
                        'eval': (n - holdout, n)}[split]
    self.rng = np.random.default_rng(seed)
    self.names = list(accounts or corpus.behaviors)
    self._cdf = {}
    action, valid = self.demos()
    self.demo_counts = dict(wait=int((valid & (action == 0)).sum()),
                            add=int((valid & (action == 1)).sum()))
    self.demo_weights = tuple(demo_weights) if demo_weights else (
        L.class_weights(action, valid, demo_balance) if weight == 'demo' else (1., 1.))

  def demos(self):
    """(action, valid) of every demonstration row of the split's accounts."""
    parts = [self.corpus.window(name, self.lo, self.hi, market=False) for name in self.names]
    return (np.concatenate([d['demo_action'] for d in parts]),
            np.concatenate([d['demo_valid'] for d in parts]))

  def _marks(self, name):
    """Per row of one account: 1 where the sampling weight counts a label."""
    on = self.corpus.labelled[name]
    if self.weight is None:
      return np.ones(len(on))
    cursor = self.corpus.behaviors[name]['cursor']
    if self.weight == 'outcome':
      return on * self.corpus.outcome['outcome_valid'][cursor]
    return on * self.corpus.demo_valid[cursor]

  def _starts(self, length):
    """Cumulative weights of every window start of every account."""
    if length not in self._cdf:
      if self.hi - self.lo < length:
        raise ValueError(f'Window of {length} rows exceeds the split')
      weights = []
      for name in self.names:
        marks = np.concatenate([[0.], np.cumsum(self._marks(name)[self.lo:self.hi])])
        start = np.arange(self.hi - self.lo - length + 1)
        weights.append(marks[start + length] - marks[np.minimum(start + self.burnin, start + length)])
      w = np.concatenate(weights)
      if not w.sum():
        raise ValueError('No labelled rows in the selected corpus windows')
      self._cdf[length] = np.cumsum(w)
    return self._cdf[length]

  def _stack(self, windows, length):
    rows = []
    for name, start in windows:
      data = self.corpus.window(name, start, start + length)
      valid, action = data.pop('demo_valid'), data['demo_action']
      data['demo_weight'] = np.where(valid, np.where(
          action == 1, self.demo_weights[1], self.demo_weights[0]), 0.).astype(np.float32)
      rows.append(data)
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    data['is_first'][:, 0] = True
    return data

  def _pick(self, batch, length):
    """`batch` (account, start) windows drawn by the sampling weight."""
    cdf = self._starts(length)
    per = self.hi - self.lo - length + 1
    picks = np.searchsorted(cdf, self.rng.random(batch) * cdf[-1], side='right')
    return [(self.names[i // per], self.lo + int(i % per)) for i in picks]

  def sample(self, batch, length):
    return self._stack(self._pick(batch, length), length)

  def batches(self, batch, length):
    """Every non-overlapping window of every account, in full batches."""
    windows = [(name, s) for name in self.names
               for s in range(self.lo, self.hi - length + 1, length)]
    for i in range(0, len(windows) - batch + 1, batch):
      yield self._stack(windows[i:i + batch], length)

  def outcome_allowed(self):
    """Allowed classes of every labelled row of the split."""
    out = []
    for name in self.names:
      d = self.corpus.window(name, self.lo, self.hi, market=False)
      out.append(d['outcome_allowed'][d['outcome_valid'] > 0])
    return np.concatenate(out)

  def summary(self):
    out = {}
    for name in self.names:
      d = self.corpus.window(name, self.lo, self.hi, market=False)
      allowed = d['outcome_allowed'][d['outcome_valid'] > 0]
      single = allowed.sum(-1) == 1
      ended = allowed.argmax(-1)
      out[name] = dict(
          rows=self.hi - self.lo, labels=len(allowed),
          **{e.lower(): int((single & (ended // L.NBINS == i)).sum())
             for i, e in enumerate(L.EVENTS)},
          open_at_h=int((single & (ended == L.OPEN_AT_H)).sum()),
          censored=int((~single).sum()),
          quality=int(d['outcome_quality_valid'].sum()),
          demo_wait=int((d['demo_valid'] & (d['demo_action'] == 0)).sum()),
          demo_add=int((d['demo_valid'] & (d['demo_action'] == 1)).sum()))
    return out


MIXES = ('uniform_contract', 'proportional_rows')


class MultiCorpusWindows:
  """Windows of several TRAIN contracts for one learning stage (C4 2.2).

  Every window first draws its contract -- mix='uniform_contract' gives each
  contract the same exposure, 'proportional_rows' weighs contracts by their
  split rows, `contract_weights` ({contract: weight}) sets them by hand --
  and then a window inside that contract exactly as CorpusWindows draws it
  (same split, sampling weight, accounts and burn-in). A window is a slice
  of one contract and never joins two contracts' rows. The demonstrations of
  every contract carry one pair of class weights from the pooled
  demonstrations of the selection (or `demo_weights`, e.g. the train
  selection's for an eval selection). `exposure` counts the sampled windows
  per contract and `last` lists the (contract, account, start) of the last
  batch.
  """

  def __init__(self, corpora, mix='uniform_contract', contract_weights=None, seed=0,
               demo_balance=1.0, demo_weights=None, **kwargs):
    if mix not in MIXES:
      raise ValueError(f'Unknown contract mix {mix}')
    self.parts = [CorpusWindows(c, demo_weights=(1., 1.), **kwargs) for c in corpora]
    self.contracts = [c.source['contract'] for c in corpora]
    self.rng = np.random.default_rng(seed)
    for part in self.parts:
      part.rng = self.rng
    action, valid = (np.concatenate(x) for x in zip(*(p.demos() for p in self.parts)))
    self.demo_counts = dict(
        wait=int((valid & (action == 0)).sum()), add=int((valid & (action == 1)).sum()),
        contracts={c: p.demo_counts for c, p in zip(self.contracts, self.parts)})
    self.demo_weights = tuple(demo_weights) if demo_weights else (
        L.class_weights(action, valid, demo_balance) if kwargs.get('weight') == 'demo'
        else (1., 1.))
    for part in self.parts:
      part.demo_weights = self.demo_weights
    if contract_weights:
      w = np.array([contract_weights[c] for c in self.contracts], np.float64)
    elif mix == 'uniform_contract':
      w = np.ones(len(self.parts))
    else:
      w = np.array([p.hi - p.lo for p in self.parts], np.float64)
    self.mix, self.p = mix, w / w.sum()
    self.exposure = dict.fromkeys(self.contracts, 0)
    self.last = []

  def sample(self, batch, length):
    which = self.rng.choice(len(self.parts), batch, p=self.p)
    windows = [(i, *self.parts[i]._pick(1, length)[0]) for i in which]
    rows = [self.parts[i]._stack([(name, start)], length) for i, name, start in windows]
    self.last = [(self.contracts[i], name, start) for i, name, start in windows]
    for contract, _, _ in self.last:
      self.exposure[contract] += 1
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}

  def batches(self, batch, length):
    """Every non-overlapping window of every contract, in full batches of
    one contract each."""
    for part in self.parts:
      yield from part.batches(batch, length)

  def outcome_allowed(self):
    return np.concatenate([p.outcome_allowed() for p in self.parts])

  def summary(self):
    return {c: dict(rows=p.hi - p.lo, accounts=p.names, labels=p.summary())
            for c, p in zip(self.contracts, self.parts)}
