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
their net PnL (the exam keeps no per-minute random equity), the always-ADD
account over its net PnL. Under a rule gate (V2.3) the report adds the
student's rule compliance, its fills inside the rule zone and its stop
sources; exams written before them show none. Writes
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
      always_net_pnl=(exam.get('always_add') or {}).get('net_pnl'),
      filled_entries=s.get('filled_entries'), rule_entries=s.get('rule_entries'),
      zone_entries=s.get('zone_entries'), rule_compliance=s.get('rule_compliance'),
      fill_in_zone=s.get('fill_in_zone'), stop_sources=s.get('stop_sources'),
      closed=s.get('closed', 0), win_rate_net=s.get('win_rate_net'), stop_stages=s.get('stop_stages'),
      hold_mean=(s.get('hold_minutes') or {}).get('mean'), move_points_mean=s.get('move_points_mean'),
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
  always = [p['always_net_pnl'] for p in parts]
  sources = {}
  for p in parts:
    for k, v in (p['stop_sources'] or {}).items():
      sources[k] = sources.get(k, 0) + v
  gated = [p for p in parts if p['rule_entries'] is not None]
  opened = sum(p['filled_entries'] for p in gated)
  whole.update(always_net_pnl=None if None in always else float(sum(always)), stop_sources=sources or None,
               rule_compliance=sum(p['rule_entries'] for p in gated) / opened if gated and opened else None,
               fill_in_zone=sum(p['zone_entries'] for p in gated) / opened if gated and opened else None)
  # V3: exits by stop stage, net win rate, holding and move, over the closed trades.
  stages, closed = {}, sum(p['closed'] for p in parts)
  for p in parts:
    for k, v in (p['stop_stages'] or {}).items():
      stages[k] = stages.get(k, 0) + v
  def weighted(key):
    pairs = [(p[key], p['closed']) for p in parts if p['closed'] and p[key] is not None]
    return sum(x * n for x, n in pairs) / sum(n for _, n in pairs) if pairs else None
  whole.update(closed=closed, stop_stages=stages or None, win_rate_net=weighted('win_rate_net'),
               hold_mean=weighted('hold_mean'), move_points_mean=weighted('move_points_mean'))
  for p in parts:
    del p['times'], p['equity']
  return dict(contracts=parts, whole=whole)


def _value(x, spec):
  """A number of the report, or a dash where the exam has none."""
  return '—' if x is None else format(x, spec)


def _sources(counts):
  return '—' if not counts else ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))


def markdown(report):
  out = ['# Paper trading', '',
         f"Contracts in time order: {', '.join(c.split('.')[-1] for c in report['contracts'])}. "
         'Each contract starts from the zero market carry and the initial balance; positions close at '
         'each main roll; the whole period chains the contracts as one account. Random: 5 accounts that '
         "ADD at the student's own ADD rate, chained per seed. Always-ADD: one account that ADDs at "
         'every legal decision (under a rule gate: every rule position it is flat at).', '']
  out += ['## Whole period', '',
          '| policy | exits (TP / SL points or strategy) | period | net PnL | random net PnL (mean ± std) | beats random seeds '
          '| always-ADD net PnL | max drawdown | months > 0 | ADDs | TP / SL | rule compliance | fills in zone '
          '| stop sources |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for tag, r in report['policies'].items():
    w, rn = r['whole'], r['whole']['random_net_pnl']
    rules = r.get('rules') or dict(take_profit_points=5, stop_loss_points=30)
    # A strategy file sets each trade's own SL / TP (V2): name it, never the V1 default distances.
    exits = (f"strategy {rules['strategy']}" if rules.get('strategy')
             else f"{rules['take_profit_points']} / {rules['stop_loss_points']}")
    out.append(f"| {tag} | {exits} "
               f"| {w['first'][:10]} → {w['last'][:10]} | {w['net_pnl']:.0f} "
               f"| {rn['mean']:.0f} ± {rn['std']:.0f} | {rn['beaten']}/{rn['seeds']} "
               f"| {_value(w.get('always_net_pnl'), '.0f')} "
               f"| {w['max_drawdown']:.0f} | {w['positive_months']}/{w['months']} | {w['adds']} "
               f"| {w['tp']} / {w['sl']} | {_value(w.get('rule_compliance'), '.3f')} "
               f"| {_value(w.get('fill_in_zone'), '.3f')} | {_sources(w.get('stop_sources'))} |")
  out += ['', '## Exits and holding', '',
          '| policy | closed trades | net win rate | stop stages | hold (market minutes, mean) '
          '| move per trade (points, mean) |', '|---|---|---|---|---|---|']
  for tag, r in report['policies'].items():
    w = r['whole']
    out.append(f"| {tag} | {w.get('closed', '—')} | {_value(w.get('win_rate_net'), '.3f')} "
               f"| {_sources(w.get('stop_stages'))} | {_value(w.get('hold_mean'), '.0f')} "
               f"| {_value(w.get('move_points_mean'), '.2f')} |")
  strategies = [(t, r['strategies']) for t, r in report['policies'].items() if r.get('strategies')]
  if strategies:
    out += ['', 'Training / paper strategy (version):']
    out += [f"- {t}: {(s['train'] or {}).get('version', '—')} / {(s['paper'] or {}).get('version', '—')}"
            for t, s in strategies]
  out += ['', '## Per contract', '',
          '| policy | contract | days | net PnL | random net PnL (mean) | always-ADD net PnL | max drawdown | ADDs '
          '| ADD rate | TP / SL | occupancy | reward/min (random) | Q flips | rule compliance | stop sources '
          '| complete |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for tag, r in report['policies'].items():
    for p in r['contracts']:
      out.append(f"| {tag} | {p['contract'].split('.')[-1]} | {p['trading_days']} | {p['net_pnl']:.0f} "
                 f"| {np.mean(p['random_net_pnl']):.0f} | {_value(p.get('always_net_pnl'), '.0f')} "
                 f"| {p['max_drawdown']:.0f} | {p['adds']} "
                 f"| {p['add_rate']:.4f} | {p['tp']} / {p['sl']} | {p['occupancy']:.3f} "
                 f"| {p['reward_per_minute']:.6f} ({np.mean(p['random_reward_per_minute']):.6f}) "
                 f"| {p['q_flip_share']:.3f} | {_value(p.get('rule_compliance'), '.3f')} "
                 f"| {_sources(p.get('stop_sources'))} | {p['complete']} |")
  out += ['', '## Monthly net PnL', '']
  tags = list(report['policies'])
  months = sorted({m for r in report['policies'].values() for m in r['whole']['monthly']})
  out += ['| month | ' + ' | '.join(tags) + ' |', '|---|' + '---|' * len(tags)]
  for m in months:
    out.append(f'| {m} | ' + ' | '.join(
        f"{report['policies'][t]['whole']['monthly'].get(m, 0.):.0f}" for t in tags) + ' |')
  out += ['', 'Q flips: share of ADD-legal decisions whose ADD/WAIT the Q anchor turns '
          '(sign of z = anchor + correction differs from the sign of the correction).',
          'Rule compliance: entries at a rule position / entries; fills in zone: entries filled inside '
          "the rule zone of their touch / entries; stop sources: the student's entries by the source of "
          'their stop (RULE / CAPPED / GAP under a rule gate). A dash: no rule gate, or an exam written '
          'before these numbers.',
          'Stop stages (V3, entry.exit_rule): the closed trades by the stage of their last stop '
          '(INITIAL / BREAKEVEN / LOCK / TROUGH); net win rate: closed trades with net PnL > 0.', '']
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
    rules = meta['source'].get('rules') or {}
    strategy = meta['source'].get('strategy')
    for c in contracts:
      source = read(Path(args.run) / t / c / 'exam.json')['source']
      if (source.get('rules') or {}) != rules or source.get('strategy') != strategy:
        raise ValueError(f'{t}: its contracts trade under different TP / SL rules or strategies')
    report['policies'][t]['strategies'] = dict(train=meta['source'].get('train_strategy'),
                                               paper=meta['source'].get('paper_strategy'))
    if strategy:
      report['policies'][t]['rules'] = dict(strategy=Path(strategy).name)
    else:
      report['policies'][t]['rules'] = dict(take_profit_points=rules.get('take_profit_points', 5),
                                            stop_loss_points=rules.get('stop_loss_points', 30))
  run = Path(args.run)
  (run / 'paper_report.json').write_text(json.dumps(report, indent=1))
  (run / 'paper_report.md').write_text(markdown(report))
  print(markdown(report))


if __name__ == '__main__':
  main()
