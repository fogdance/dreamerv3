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

A window starts from the zero market state with a loss burn-in, or, given a
MarketCarry store of its contract, from the causal market carry the policy
holds at its first cursor on the continuous path (user decision 2026-10-24
after C4 review R3: zero-carry windows moved pi_0000's ADD decisions).
"""
import json
from pathlib import Path

import ml_dtypes  # noqa: F401 (numpy knows the bfloat16 carry dtype)
import numpy as np

from gym_trading_env.research import ts_v1_corpus as C

from . import labels as L

ACCOUNT_KEYS = ('position_state', 'action_mask', 'account_native', 'is_last',
                'is_terminal')
LABEL_ACCOUNT = 'P0'
CARRY_KEYS = ('deter', 'stoch')


class MarketCarry:
  """The causal market carry of one contract (run.market_carry): row t holds
  the market RSSM state the policy has before the minute of cursor t on its
  continuous path from the first market minute. It holds for every policy
  with the same market brain and policy seed counter: the market brain never
  reads the account or the actions, and L2-L5 never train it. Windows read
  it under the keys 'carry/deter' and 'carry/stoch' (memory-mapped)."""

  def __init__(self, folder):
    self.folder = Path(folder)
    self.meta = json.loads((self.folder / 'meta.json').read_text())
    dtype = np.dtype(self.meta['dtype'])
    self.arrays = {k: np.load(self.folder / f'{k}.npy', mmap_mode='r').view(dtype)
                   for k in CARRY_KEYS}

  @staticmethod
  def write(folder, arrays, meta):
    """The arrays as raw bits (numpy has no bfloat16 file format), then
    meta.json last: a folder with a meta.json holds a whole store."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for k in CARRY_KEYS:
      bits = arrays[k].view(np.dtype(f'u{arrays[k].dtype.itemsize}'))
      np.save(folder / f'{k}.npy', bits)
    tmp = folder / 'meta.json.tmp'
    tmp.write_text(json.dumps(dict(meta, dtype=str(arrays['deter'].dtype)), indent=1))
    tmp.rename(folder / 'meta.json')

  def at(self, cursors):
    """{'carry/<key>': [len(cursors), ...]}: the carry before each cursor."""
    cursors = np.asarray(cursors)
    return {f'carry/{k}': np.asarray(v[cursors]) for k, v in self.arrays.items()}


def pop_carry(data):
  """The 'carry/*' arrays of a batch of windows ({} when they start from the
  zero state), removed from `data` so the rest matches the agent's inputs."""
  return {k.split('/', 1)[1]: data.pop(k) for k in list(data) if k.startswith('carry/')}


class TSCorpus:
  """One contract's teaching material, loaded once and shared by windows.
  The market arrays (most of the material) are memory-mapped: windows read
  their rows, and many contracts share the page cache instead of each
  holding a full copy (C4 2.1). `carry_root` holds the contract's
  MarketCarry: its windows then start from the causal market carry."""

  def __init__(self, folder, behaviors=None, carry_root=None):
    self.folder = Path(folder)
    self.source = json.loads((self.folder / 'source.json').read_text())
    self.market = C.load_arrays(self.folder / 'market', mmap='r')
    rows = self.source['rows']
    for k in C.MARKET_KEYS:
      _table_check(f'market/{k}', self.market[k], rows)
    self.carry = None
    if carry_root is not None:
      self.carry = MarketCarry(Path(carry_root) / self.source['contract'])
      for k, v in self.carry.arrays.items():
        _table_check(f'carry/{k}', v, rows)
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
  not a data break; its first `burnin` rows carry no label loss. With the
  corpus' MarketCarry it starts instead from the stored causal carry at its
  first cursor ('carry/*', is_first only at the first market minute), and no
  row needs a burn-in.
  weight='outcome' / 'demo' samples windows in proportion to their labelled
  rows after the burn-in, weight=None uniformly. `accounts` selects the
  behavior accounts (default: all). Demonstrations carry class weights
  (`L.class_weights` over the split's demonstrations unless `demo_weights`
  is given, e.g. the train split's for an eval split).

  `remedial` lists error cursors of S5 feedback (B2 remediation): a share
  `remedial_fraction` of every batch (rounded) are windows drawn uniformly
  around a uniformly drawn error row of an account, so that the row is in the
  window after the burn-in; the rest of the batch is drawn by the natural
  weight. An error row must carry a label of the sampling weight; its window
  and labels are exactly the natural ones. `curriculum()` counts the
  labelled rows the windows showed (distinct and repeated) and the
  remediation exposure.
  """

  def __init__(self, corpus, split='all', weight=None, burnin=0, holdout=2048,
               seed=0, demo_balance=1.0, demo_weights=None, accounts=None, remedial=None,
               remedial_fraction=0.):
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
    self._cdf, self._marked = {}, {}
    action, valid = self.demos()
    self.demo_counts = dict(wait=int((valid & (action == 0)).sum()),
                            add=int((valid & (action == 1)).sum()))
    self.demo_weights = tuple(demo_weights) if demo_weights else (
        L.class_weights(action, valid, demo_balance) if weight == 'demo' else (1., 1.))
    if not 0 <= remedial_fraction <= 1:
      raise ValueError(f'Remedial fraction {remedial_fraction} is not in [0, 1]')
    self.remedial_fraction = float(remedial_fraction)
    self.remedial, self.remedial_dropped = self._remedial_rows(remedial)
    self.seen = {name: np.zeros(n, bool) for name in self.names}
    self.windows = self.remedial_windows = self.label_rows = 0
    self.remedial_seen = set()

  def _remedial_rows(self, cursors):
    """(account, row) of every error cursor of each account inside the split,
    and how many cursors fall outside it (holdout, or before the burn-in)."""
    if cursors is None:
      return [], 0
    cursors, out = np.unique(np.asarray(cursors, np.int64)), []
    for name in self.names:
      cursor = self.corpus.behaviors[name]['cursor']
      rows = np.flatnonzero(np.isin(cursor, cursors))
      if len(rows) != len(cursors):
        raise ValueError(f'{name}: remedial cursors outside the corpus rows')
      marks = self._marks(name)
      if not (marks[rows] > 0).all():
        bad = cursor[rows[marks[rows] <= 0]]
        raise ValueError(f'{name}: remedial cursors carry no {self.weight} label, first {bad[0]}')
      inside = (rows >= self.lo + self.burnin) & (rows < self.hi)
      out += [(name, int(r)) for r in rows[inside]]
    return out, int(len(cursors) * len(self.names) - len(out))

  def remedial_count(self, batch):
    """Remedial windows of a batch: round(fraction x batch), none without
    error rows."""
    return int(round(self.remedial_fraction * batch)) if self.remedial else 0

  def _pick_remedial(self, n, length):
    """`n` windows, each around an error row drawn uniformly: its start is
    drawn uniformly among the windows that hold the row after the burn-in."""
    if not n:
      return []
    if self.hi - self.lo < length:
      raise ValueError(f'Window of {length} rows exceeds the split')
    out = []
    for i in self.rng.integers(len(self.remedial), size=n):
      name, row = self.remedial[i]
      lo, hi = max(self.lo, row - length + 1), min(row - self.burnin, self.hi - length)
      out.append((name, int(self.rng.integers(lo, hi + 1)), row))
    return out

  def _count(self, windows, length, remedial=()):
    """Record sampled (account, start) windows; `remedial` holds the error
    rows of the remedial ones."""
    for name, start in windows:
      shown = slice(start + self.burnin, start + length)
      self.seen[name][shown] = True
      self.label_rows += int((self._marks(name)[shown] > 0).sum())
    self.windows += len(windows)
    self.remedial_windows += len(remedial)
    self.remedial_seen.update(remedial)

  def curriculum(self):
    """Labelled rows the sampled windows showed after the burn-in: distinct
    market cursors (one per row of every account), repeats (label_rows) and
    the remediation exposure."""
    unique = sum(int((self.seen[name] & (self._marks(name) > 0)).sum()) for name in self.names)
    return dict(windows=self.windows, remedial_windows=self.remedial_windows,
                remedial_fraction=self.remedial_fraction, remedial_rows=len(self.remedial),
                remedial_rows_dropped=self.remedial_dropped,
                remedial_rows_seen=len(self.remedial_seen), unique_label_rows=unique,
                label_rows=self.label_rows)

  def demos(self):
    """(action, valid) of every demonstration row of the split's accounts."""
    parts = [self.corpus.window(name, self.lo, self.hi, market=False) for name in self.names]
    return (np.concatenate([d['demo_action'] for d in parts]),
            np.concatenate([d['demo_valid'] for d in parts]))

  def _marks(self, name):
    """Per row of one account: 1 where the sampling weight counts a label."""
    if name not in self._marked:
      on = self.corpus.labelled[name]
      cursor = self.corpus.behaviors[name]['cursor']
      self._marked[name] = (
          np.ones(len(on)) if self.weight is None else
          on * self.corpus.outcome['outcome_valid'][cursor] if self.weight == 'outcome' else
          on * self.corpus.demo_valid[cursor])
    return self._marked[name]

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
    if self.corpus.carry is None:
      data['is_first'][:, 0] = True
    else:
      data.update(self.corpus.carry.at(
          [self.corpus.behaviors[name]['cursor'][start] for name, start in windows]))
    return data

  def _pick(self, batch, length):
    """`batch` (account, start) windows drawn by the sampling weight."""
    cdf = self._starts(length)
    per = self.hi - self.lo - length + 1
    picks = np.searchsorted(cdf, self.rng.random(batch) * cdf[-1], side='right')
    return [(self.names[i // per], self.lo + int(i % per)) for i in picks]

  def sample(self, batch, length):
    k = self.remedial_count(batch)
    remedial = self._pick_remedial(k, length)
    windows = [(name, start) for name, start, _ in remedial] + self._pick(batch - k, length)
    self._count(windows, length, [(name, row) for name, _, row in remedial])
    return self._stack(windows, length)

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
EXPOSURE_NOTE = ('remediation repeats TRAIN cursors of LABEL_ACCOUNT rows: windows and '
                 'label_rows are optimizer exposure, unique_label_rows distinct market cursors '
                 'of each account (C3 EDU-R10)')


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

  `remedial` ({contract: error cursors}, S5 feedback) makes a share
  `remedial_fraction` of every batch remedial windows (CorpusWindows): each
  draws an error row uniformly over the error rows of every contract, so a
  contract with more errors gets more of them, and the rest of the batch
  follows the contract mix.
  """

  def __init__(self, corpora, mix='uniform_contract', contract_weights=None, seed=0,
               demo_balance=1.0, demo_weights=None, remedial=None, remedial_fraction=0.,
               **kwargs):
    if mix not in MIXES:
      raise ValueError(f'Unknown contract mix {mix}')
    self.contracts = [c.source['contract'] for c in corpora]
    unknown = set(remedial or ()) - set(self.contracts)
    if unknown:
      raise ValueError(f'Remedial cursors of contracts outside the selection: {sorted(unknown)}')
    self.parts = [CorpusWindows(c, demo_weights=(1., 1.), remedial=(remedial or {}).get(name),
                                remedial_fraction=remedial_fraction, **kwargs)
                  for c, name in zip(corpora, self.contracts)]
    self.remedial_fraction = float(remedial_fraction)
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
    errors = np.array([len(p.remedial) for p in self.parts], np.float64)
    k = int(round(self.remedial_fraction * batch)) if errors.sum() else 0
    windows = []
    for i in (self.rng.choice(len(self.parts), k, p=errors / errors.sum()) if k else ()):
      name, start, row = self.parts[i]._pick_remedial(1, length)[0]
      self.parts[i]._count([(name, start)], length, [(name, row)])
      windows.append((i, name, start))
    for i in self.rng.choice(len(self.parts), batch - k, p=self.p):
      name, start = self.parts[i]._pick(1, length)[0]
      self.parts[i]._count([(name, start)], length)
      windows.append((i, name, start))
    rows = [self.parts[i]._stack([(name, start)], length) for i, name, start in windows]
    self.last = [(self.contracts[i], name, start) for i, name, start in windows]
    for contract, _, _ in self.last:
      self.exposure[contract] += 1
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}

  def curriculum(self):
    """What the sampled windows showed, per contract and in total (S5 B2)."""
    parts = {c: p.curriculum() for c, p in zip(self.contracts, self.parts)}
    total = {k: sum(v[k] for v in parts.values()) for k in next(iter(parts.values()))
             if k != 'remedial_fraction'}
    return dict(total, remedial_fraction=self.remedial_fraction,
                remedial_share=total['remedial_windows'] / max(total['windows'], 1),
                contracts=parts, exposure_note=EXPOSURE_NOTE)

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
