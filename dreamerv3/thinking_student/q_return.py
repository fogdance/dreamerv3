"""Economic Q return of a CF folder (E2 R2, user 2026-10-30; e2-r2.md).

The CF engine labels the ADD/WAIT pairs with the Native Quality reward,
discounted 332/333 per market minute. NET_PNL_H2760 relabels them with the
economic value: the undiscounted net PnL, in units of one standard SL30 net
loss (1818 CNY), of every trade that is open within the next H = 2760 market
minutes:

  G(t) = sum over trades j with exit decision x_j >= t and entry decision e_j
         at advance(t, e_j) < H of net_trade_pnl_j / 1818

A trade still open when the window ends counts its realized net PnL, not 0
(its unrealized loss is never dropped): a coupled branch is flat at its
coupling cursor c and the reference runs to its true terminal, so every such
trade has its final PnL. A branch is its own stored trades before c, then
the reference trades that close at or after c.

The checks of E2 run on every contract and stop on failure: the reference
reproduces the stored R0 return, each SL penalty sits at exit_cursor - 1,
the trades sum to the account net PnL, and for every coupled pair branch PnL
minus reference PnL before c is the stored equity gap.

  python -m dreamerv3.thinking_student.q_return --collection DIR --cf DIR --out DIR
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from gym_trading_env.research import ts_v1_cf as CF
from gym_trading_env.research import ts_v1_corpus as C

NATIVE, NET_PNL = 'native', 'net_pnl_h2760'
SCALE, HORIZON = 1818., 2760
TIE = 1e-9


def _decisions(trades):
  """(entry decision, closing decision, net PnL / SCALE) of each trade."""
  return [(int(t['entry_cursor']) - 1, int(t['exit_cursor']) - 1, float(t['net_trade_pnl']) / SCALE)
          for t in trades]


def window_returns(clock, start, n, trades, horizon=HORIZON):
  """G(t) of every row t = start..start+n-1: trade j counts on the rows
  lo_j <= t <= x_j, lo_j the first row whose clock is above clock(e_j) - H."""
  rows = clock.prefix[start + np.arange(n) + 1]
  acc = np.zeros(n + 1)
  for e, x, v in _decisions(trades):
    if not 0 <= x - start < n:
      raise ValueError(f'Trade exit decision {x} outside the decision rows')
    lo = int(np.searchsorted(rows, clock.prefix[e + 1] - horizon, side='right'))
    if lo <= x - start:
      acc[lo] += v
      acc[x - start + 1] -= v
  return np.cumsum(acc)[:n]


def _branch(clock, t, decisions, horizon, closed_only=False):
  """The value at t of trades (see _decisions) that are open in the window;
  closed_only drops the trades still open at its end (only to count where
  that would change a label)."""
  now = clock.prefix[t + 1]
  return float(sum(v for e, x, v in decisions
                   if x >= t and clock.prefix[e + 1] - now < horizon
                   and not (closed_only and clock.prefix[x + 1] - now >= horizon)))


def relabel(collection, cf, horizon=HORIZON):
  """The NET_PNL pairs, targets and meta of one contract's CF, after the checks."""
  collection, cf = Path(collection), Path(cf)
  meta = json.loads((cf / 'meta.json').read_text())
  if Path(meta['collection']).resolve() != collection.resolve():
    raise ValueError(f'{cf} holds the CF of {meta["collection"]}, not of {collection}')
  pairs = [json.loads(line) for line in (cf / 'pairs.jsonl').open()]
  targets = dict(np.load(cf / 'targets.npz'))
  rows = [json.loads(line) for f in sorted((collection / 'steps').glob('*.jsonl')) for line in f.open()]
  r0 = np.concatenate([np.load(f)['reward'] for f in sorted((collection / 'replay').glob('*.npz'))])
  with (collection / 'state.pkl').open('rb') as f:
    trades = pickle.load(f)['run']['env']['attributions']
  clock = C.corpus_env(meta['corpus'], market=False).market_clock_counter
  start, n, name = rows[0]['cursor'], len(rows), cf.name
  cursors = np.array([r['cursor'] for r in rows])
  if not np.array_equal(cursors, start + np.arange(n)) or n != meta['reference_rows']:
    raise ValueError(f'{name}: reference rows')
  action = np.array([r['executed'] for r in rows])

  g0 = CF.discounted_returns(clock, cursors, r0)
  if not np.allclose(g0, targets['q_real'], atol=1e-4, rtol=0):
    raise ValueError(f'{name}: the R0 reference return does not reproduce q_real')
  if any(r0[int(t['exit_cursor']) - 1 - start] > -2.9 for t in trades if t['reason'] == 'STOP_LOSS'):
    raise ValueError(f'{name}: an SL penalty is not at exit_cursor - 1')
  ref = _decisions(trades)
  pnl = sum(float(t['net_trade_pnl']) for t in trades)
  if abs(pnl - rows[-1]['account_net_pnl']) > 1e-6:
    raise ValueError(f'{name}: trades {pnl} vs account {rows[-1]["account_net_pnl"]}')
  g = window_returns(clock, start, n, trades, horizon)

  out, open_at_end = [], 0
  for p in pairs:
    q = dict(p)
    out.append(q)
    if p['status'] != 'VALID_COMPLETE' or p['delta_g'] is None:
      continue
    t, own = p['cursor'], int(action[p['cursor'] - start])
    if own != p['reference_action']:
      raise ValueError(f'{name}@{t}: reference action')
    if not np.isclose(p['g_add' if own == 1 else 'g_wait'], g0[t - start], atol=1e-6, rtol=0):
      raise ValueError(f'{name}@{t}: the stored own return differs from the R0 reference')
    if p['reason'] == 'COUPLED':
      c = p['coupled_at']
      before = sum(v for _, x, v in ref if t <= x < c) * SCALE
      own_pnl = sum(float(b['net_trade_pnl']) for b in p['trades'])
      if abs((own_pnl - before) - p['equity_gap']) > 1e-6:
        raise ValueError(f'{name}@{t}: branch PnL - reference PnL != equity gap '
                         f'({own_pnl} - {before} vs {p["equity_gap"]})')
      tail = [d for d in ref if d[1] >= c]
    elif p['reason'] == 'TERMINAL':
      tail = []
    else:
      raise ValueError(f'{name}@{t}: a valid pair neither coupled nor terminal')
    own_trades = [*_decisions(p['trades']), *tail]
    branch = _branch(clock, t, own_trades, horizon)
    mine = float(g[t - start])
    # pairs whose label holds a trade still open at the window end on one side
    dropped = _branch(clock, t, own_trades, horizon, True) - _branch(clock, t, ref, horizon, True)
    open_at_end += abs((branch - mine) - dropped) > TIE
    q['g'] = branch
    q['g_add'], q['g_wait'] = (mine, branch) if own == 1 else (branch, mine)
    q['delta_g'] = q['g_add'] - q['g_wait']

  tg = dict(targets)
  tg['q_real'] = g.astype(np.float32)
  tg['q_cf'] = np.zeros(n, np.float32)
  valid = [q for q in out if q['status'] == 'VALID_COMPLETE' and q['delta_g'] is not None]
  for q in valid:
    tg['q_cf'][q['cursor'] - start] = q['g']
  if not np.array_equal(tg['q_cf_valid'] > 0, targets['q_cf_valid'] > 0):
    raise ValueError(f'{name}: the valid CF rows changed')
  dg = np.array([q['delta_g'] for q in valid])
  meta = dict(meta, delta_g=dict(mean=float(dg.mean()), std=float(dg.std())) if len(dg) else None,
              q_return=dict(name=NET_PNL, scale=SCALE, horizon=horizon, discount=None,
                            relabeled_from=str(cf.resolve()), native_delta_g=meta['delta_g'],
                            stats=stats(valid, open_at_end)))
  return dict(meta=meta, pairs=out, targets=tg)


def stats(valid, open_at_end=0):
  """The relabeled labels by coupling kind: a pair coupled at the EXACT state
  has the same funds on both paths, so its Delta G is 0 when both close
  every trade by the coupling (e2-r2.md asks to watch this share)."""
  def one(pairs):
    dg = np.array([q['delta_g'] for q in pairs])
    if not len(dg):
      return dict(pairs=0)
    return dict(pairs=len(dg), add_better=float((dg > TIE).mean()), tie=float((np.abs(dg) <= TIE).mean()),
                wait_better=float((dg < -TIE).mean()), mean=float(dg.mean()), std=float(dg.std()))
  kinds = {}
  for q in valid:
    kinds.setdefault('TERMINAL' if q['reason'] == 'TERMINAL' else q.get('coupling_kind') or 'UNKNOWN',
                     []).append(q)
  return dict(one(valid), open_at_window_end=open_at_end,
              by_kind={k: one(v) for k, v in sorted(kinds.items())})


def write(res, out):
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  np.savez(out / 'targets.npz', **res['targets'])
  with (out / 'pairs.jsonl').open('w') as f:
    f.writelines(json.dumps(q) + '\n' for q in res['pairs'])
  (out / 'meta.json').write_text(json.dumps(res['meta'], indent=1))  # last: the folder is complete


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--collection', required=True)
  ap.add_argument('--cf', required=True, help='the native CF folder')
  ap.add_argument('--out', required=True)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  a = ap.parse_args(argv)
  res = relabel(a.collection, a.cf, a.horizon)
  write(res, a.out)
  print(json.dumps(res['meta']['q_return'], indent=1))


if __name__ == '__main__':
  main()
