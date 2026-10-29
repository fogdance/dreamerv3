"""Paper-trading report (read only): the PAPER exams of frozen policies on
the contracts of the paper-trading pool, chained in time order as one
account (user 2026-10-29).

  python -m dreamerv3.thinking_student.paper --run DIR --pool POOL TAG [TAG ...]

Each TAG is a folder of DIR holding one exam folder per contract
(<DIR>/<TAG>/<contract>/exam.json and its collect/steps). Every contract
starts from the zero market carry at its first minute, as in training and
DEV, and from the initial balance; positions are closed at each main roll,
so the chained equity is the previous contracts' net PnL plus this
contract's equity change. Random-entry accounts are chained per seed over
their net PnL (the exam keeps no per-minute random equity). Writes
<DIR>/paper_report.json and <DIR>/paper_report.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
  return json.loads(Path(path).read_text())


def steps(folder):
  return [json.loads(line) for f in sorted((Path(folder) / 'collect' / 'steps').glob('*.jsonl'))
          for line in f.open()]


def contract_result(folder):
  """One contract's exam numbers and per-decision (time, equity) of the student."""
  exam, rows = read(Path(folder) / 'exam.json'), steps(folder)
  if exam['label'] != 'PAPER' or not exam['teacher_free'] or exam['trained']:
    raise ValueError(f'{folder}: not a Teacher-free PAPER exam')
  s, rnd = exam['student'], exam['random']
  legal = [r for r in rows if r['mask'][1] > 0.5]
  # Decisions whose ADD/WAIT the Q anchor turns (z = anchor + correction).
  flips = sum((r['z'] > 0) != (r['correction'] > 0) for r in legal)
  return dict(
      contract=exam['source']['row']['contract'], complete=exam['complete'],
      first=rows[0]['time'], last=rows[-1]['time'], minutes=s['minutes'],
      trading_days=s['trading_days'], adds=s['adds'], add_rate=s['add_rate'], tp=s['tp'], sl=s['sl'],
      open_trades=s['open_trades'], occupancy=s['occupancy'], reward_per_minute=s['reward_per_minute'],
      net_pnl=s['net_pnl'], max_drawdown=s['max_drawdown'],
      random_net_pnl=[x['net_pnl'] for x in rnd['per_seed']],
      random_reward_per_minute=[x['reward_per_minute'] for x in rnd['per_seed']],
      q_flip_share=flips / max(len(legal), 1),
      initial=rows[0]['equity'], times=[r['time'] for r in rows],
      equity=[r['equity'] for r in rows])


def chain(parts):
  """The contracts as one account: times, chained equity, monthly PnL."""
  times, equity, offset = [], [], 0.
  for p in parts:
    times += p['times']
    equity += [offset + e for e in p['equity']]
    offset += p['net_pnl']
  equity = np.array(equity, np.float64)
  months, last = {}, equity[0]
  for t, e in zip(times, equity):
    months[t[:7]] = e
  monthly = {}
  for m, e in months.items():
    monthly[m], last = float(e - last), e
  # The last contract's own end: its net PnL already counts the final equity.
  final = parts[0]['initial'] + offset
  monthly[times[-1][:7]] += float(final - equity[-1])
  return dict(net_pnl=float(offset), max_drawdown=float((np.maximum.accumulate(equity) - equity).max()),
              monthly=monthly, positive_months=sum(v > 0 for v in monthly.values()),
              months=len(monthly))


def policy_report(folder, contracts):
  parts = [contract_result(Path(folder) / c) for c in contracts]
  whole = chain(parts)
  seeds = np.array([p['random_net_pnl'] for p in parts]).sum(0)
  whole.update(random_net_pnl=dict(mean=float(seeds.mean()), std=float(seeds.std()),
                                   min=float(seeds.min()), max=float(seeds.max()),
                                   beaten=int((whole['net_pnl'] > seeds).sum()), seeds=len(seeds)),
               first=parts[0]['first'], last=parts[-1]['last'],
               minutes=sum(p['minutes'] for p in parts),
               trading_days=sum(p['trading_days'] for p in parts),
               adds=sum(p['adds'] for p in parts), tp=sum(p['tp'] for p in parts),
               sl=sum(p['sl'] for p in parts))
  for p in parts:
    del p['times'], p['equity']
  return dict(contracts=parts, whole=whole)


def markdown(report):
  out = ['# Paper trading', '',
         f"Contracts in time order: {', '.join(c.split('.')[-1] for c in report['contracts'])}. "
         'Each contract starts from the zero market carry and the initial balance; positions close at '
         'each main roll; the whole period chains the contracts as one account. Random: 5 accounts that '
         "ADD at the student's own ADD rate, chained per seed.", '']
  out += ['## Whole period', '',
          '| policy | period | net PnL | random net PnL (mean ± std) | beats random seeds | max drawdown '
          '| months > 0 | ADDs | TP / SL |', '|---|---|---|---|---|---|---|---|---|']
  for tag, r in report['policies'].items():
    w, rn = r['whole'], r['whole']['random_net_pnl']
    out.append(f"| {tag} | {w['first'][:10]} → {w['last'][:10]} | {w['net_pnl']:.0f} "
               f"| {rn['mean']:.0f} ± {rn['std']:.0f} | {rn['beaten']}/{rn['seeds']} "
               f"| {w['max_drawdown']:.0f} | {w['positive_months']}/{w['months']} | {w['adds']} "
               f"| {w['tp']} / {w['sl']} |")
  out += ['', '## Per contract', '',
          '| policy | contract | days | net PnL | random net PnL (mean) | max drawdown | ADDs | ADD rate '
          '| TP / SL | occupancy | reward/min (random) | Q flips | complete |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for tag, r in report['policies'].items():
    for p in r['contracts']:
      out.append(f"| {tag} | {p['contract'].split('.')[-1]} | {p['trading_days']} | {p['net_pnl']:.0f} "
                 f"| {np.mean(p['random_net_pnl']):.0f} | {p['max_drawdown']:.0f} | {p['adds']} "
                 f"| {p['add_rate']:.4f} | {p['tp']} / {p['sl']} | {p['occupancy']:.3f} "
                 f"| {p['reward_per_minute']:.6f} ({np.mean(p['random_reward_per_minute']):.6f}) "
                 f"| {p['q_flip_share']:.3f} | {p['complete']} |")
  out += ['', '## Monthly net PnL', '']
  tags = list(report['policies'])
  months = sorted({m for r in report['policies'].values() for m in r['whole']['monthly']})
  out += ['| month | ' + ' | '.join(tags) + ' |', '|---|' + '---|' * len(tags)]
  for m in months:
    out.append(f'| {m} | ' + ' | '.join(
        f"{report['policies'][t]['whole']['monthly'].get(m, 0.):.0f}" for t in tags) + ' |')
  out += ['', 'Q flips: share of ADD-legal decisions whose ADD/WAIT the Q anchor turns '
          '(sign of z = anchor + correction differs from the sign of the correction).', '']
  return '\n'.join(out)


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--run', required=True, help='paper-trading folder with one folder per TAG')
  parser.add_argument('--pool', required=True, help='paper-trading pool (contract order)')
  parser.add_argument('tags', nargs='+')
  args = parser.parse_args(argv)
  rows = read(args.pool)['files']
  contracts = [r['contract'] for r in sorted(rows, key=lambda r: r['first_timestamp'])]
  report = dict(pool=str(Path(args.pool).resolve()), contracts=contracts,
                policies={t: policy_report(Path(args.run) / t, contracts) for t in args.tags})
  for t in args.tags:
    meta = read(Path(args.run) / t / contracts[0] / 'exam.json')
    report['policies'][t]['policy'] = meta['policy']
  run = Path(args.run)
  (run / 'paper_report.json').write_text(json.dumps(report, indent=1))
  (run / 'paper_report.md').write_text(markdown(report))
  print(markdown(report))


if __name__ == '__main__':
  main()
