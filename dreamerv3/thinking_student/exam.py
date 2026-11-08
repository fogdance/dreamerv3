"""Thinking Student V1 Teacher-free exam (C4 5): a frozen complete policy
trades a Native account on its own, and random-entry accounts trade the same
market minutes under the same Native rules.

  python -m dreamerv3.thinking_student.exam --policy DIR --out DIR \\
      --pool configs/thinking_student/markets/pool.json --contract CZCE.FG409 \
      [--strategy configs/thinking_student/markets/strategy_v2_short_structural.yaml]
  python -m dreamerv3.thinking_student.exam --policy DIR --out DIR --corpus DIR --steps N
  python -m dreamerv3.thinking_student.exam --policy DIR --out DIR \\
      --paper-pool configs/thinking_student/markets/paper_pool.json --contract CZCE.FG501 \\
      [--take-profit-points 5] [--stop-loss-points 30 | --strategy STRATEGY]

DEV          a VALIDATION row of the TS market pool (gym ts_v1_corpus.dev_row),
             its env inputs only (write_inputs, as for the TRAIN corpora): an
             independent market under the same Native rules (--strategy: the
             V2 strategy the TRAIN corpora were built with).
PAPER        a row of the paper-trading pool (role 'paper', never eligible for
             training; user 2026-10-29): paper trading on later markets to see
             the policy's profit. Like DEV it only reads env inputs; it is not
             used to choose or tune a policy. Paper trading may set other
             TP / SL distances than the trained TP5 / SL30 (user 2026-10-30);
             the student and the random accounts then trade under them.
             --strategy: the V2 strategy of a V2 policy's corpora (user
             2026-11-04), in place of any TP / SL distances.
TRAIN_SMOKE  the env of a TRAIN corpus, usually cut short: an engineering
             run, never an out-of-sample result.
DEV_NOT_CONFIGURED  written by the orchestrator when no DEV market is set.

The student path is run.collect of the frozen policy from the first market
minute (market memory runs on, never reset). Each random account ADDs at a
legal decision with probability = the student's ADDs / its legal decisions
(gym Behavior 'probabilistic'), one account per seed over the same minutes;
no seed is dropped for its outcome. One more account ADDs at every legal
decision (gym Behavior 'always_add'). Teachers, facts and CF never enter,
nothing is trained, and TEST is never read.

Under a rule gate (V2.3 strategy entry.rule_gate, user 2026-11-05: "规则决定
学生能在哪里交易；大脑决定在这些位置中什么时候交易") an ADD is legal only at
a rule position, so the random accounts time their entries at random inside
the rule and the always-ADD account enters at every rule position it is flat
at. Every account reports its entries by stop source, the share of them at a
rule position (rule_compliance; below 1 under a gate it is a bug and the exam
fails) and the share filled inside the rule zone of their touch.
"""
import argparse
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DEFINITIONS = dict(
    minutes='decisions: one per market minute of the Native env from its first minute',
    trading_days='official trading days of the decided minutes',
    reward='Native Quality TP reward summed over the decisions',
    reward_per_minute='reward / minutes',
    reward_per_day='reward / trading_days',
    occupancy='minutes holding a position (entry to exit, or to the last decision) / minutes',
    max_drawdown='largest fall of account equity (at each decision) below its running peak, CNY',
    net_pnl='equity at the last decision - initial balance, CNY',
    tp_hold_minutes='minutes from entry to exit of TAKE_PROFIT trades',
    random_probability="the student's ADDs / its legal decisions",
    always_add='one account that ADDs at every legal decision (under a rule gate: every rule position '
               'it is flat at)',
    filled_entries='entries (ENTRY fills) of the account',
    stop_sources='entries by the source of their stop: RULE / CAPPED / GAP under a rule gate, '
                 'STRUCTURAL / CAPPED / NO_LINE under the stateless stop rule, NONE for fixed TP / SL',
    rule_compliance='entries at a rule position (the fill names its touch) / entries; None without a '
                    'rule gate or without entries',
    fill_in_zone='entries filled inside the rule zone of their touch ([H-zone, H] SHORT, [L, L+zone] '
                 'LONG) / entries; None without a rule gate or without entries',
    win_rate_net='closed trades with net PnL (both fees) > 0 / closed trades',
    stop_stages='closed trades by the stage of their stop under entry.exit_rule (V3): INITIAL / '
                'BREAKEVEN / LOCK / TROUGH; None without one',
    hold_minutes='market minutes from entry to exit of the closed trades',
    move_points_mean='mean exit - entry move of the closed trades in points (SHORT: entry - exit)')
SCORES = ('reward', 'reward_per_minute', 'reward_per_day', 'net_pnl', 'max_drawdown')


def metrics(rows, trades, entries, days, initial, fills=(), zones=None):
  """One account's exam numbers. rows: per decision dict(cursor, legal,
  executed, reward, equity); trades: closed Native attributions; entries:
  positions still open after the last decision; days: official trading day
  per cursor; fills: the account's Native fills; zones: the rule zone in
  points per side under a rule gate (None without one)."""
  n = len(rows)
  opened = [f for f in fills if f['reason'] == 'ENTRY']
  sources = {}
  for f in opened:
    sources[f.get('stop_source', 'NONE')] = sources.get(f.get('stop_source', 'NONE'), 0) + 1
  rule = [f for f in opened if f.get('rule_event') is not None]
  if zones is not None and len(rule) != len(opened):
    raise ValueError(f'a rule-gated account has {len(opened) - len(rule)} entries off its rule positions')
  # Inside the zone: at or below the line by at most `zone` (SHORT), at or above it (LONG).
  inside = sum(0 <= (1 if f['side'] == 'SHORT' else -1) * (float(f['line']) - float(f['price']))
               <= zones[f['side']] for f in rule) if zones is not None else None
  reward = float(sum(r['reward'] for r in rows))
  covered = len({days[r['cursor']] for r in rows})
  legal = sum(bool(r['legal']) for r in rows)
  adds = sum(r['executed'] == 1 for r in rows)
  exits = {}
  for t in trades:
    exits[t['reason']] = exits.get(t['reason'], 0) + 1
  last = rows[-1]['cursor']
  held = (sum(t['exit_cursor'] - t['entry_cursor'] for t in trades)
          + sum(last - e['entry_cursor'] for e in entries))
  tp = np.array([t['exit_cursor'] - t['entry_cursor'] for t in trades
                 if t['reason'] == 'TAKE_PROFIT'], np.float64)
  equity = np.array([r['equity'] for r in rows], np.float64)
  # V3 long trend (entry.exit_rule): exits by the stage of their stop, net wins, holding.
  stages = {}
  for t in trades:
    if 'stop_stage' in t:
      stages[t['stop_stage']] = stages.get(t['stop_stage'], 0) + 1
  hold = np.array([t['exit_cursor'] - t['entry_cursor'] for t in trades], np.float64)
  move = np.array([(1 if t['side'] == 'LONG' else -1) * (float(t['exit_price']) - float(t['entry_price']))
                   for t in trades], np.float64)
  net = np.array([float(t['net_trade_pnl']) for t in trades if 'net_trade_pnl' in t], np.float64)
  return dict(
      minutes=n, trading_days=covered, reward=reward, reward_per_minute=reward / n,
      reward_per_day=reward / max(covered, 1), legal=legal, adds=adds,
      add_rate=adds / max(legal, 1), closed=len(trades), exits=exits,
      tp=exits.get('TAKE_PROFIT', 0), sl=exits.get('STOP_LOSS', 0),
      other_exits=len(trades) - exits.get('TAKE_PROFIT', 0) - exits.get('STOP_LOSS', 0),
      open_trades=len(entries), occupancy=held / n,
      tp_hold_minutes=dict(mean=float(tp.mean()), median=float(np.median(tp)),
                           p90=float(np.percentile(tp, 90))) if len(tp) else None,
      max_drawdown=float((np.maximum.accumulate(equity) - equity).max()),
      net_pnl=float(equity[-1] - initial), filled_entries=len(opened), stop_sources=sources,
      rule_entries=None if zones is None else len(rule), zone_entries=inside,
      rule_compliance=len(rule) / len(opened) if zones is not None and opened else None,
      fill_in_zone=inside / len(opened) if zones is not None and opened else None,
      win_rate_net=float((net > 0).mean()) if len(net) else None,
      stop_stages=stages or None,
      hold_minutes=dict(mean=float(hold.mean()), median=float(np.median(hold)),
                        p90=float(np.percentile(hold, 90))) if len(hold) else None,
      move_points_mean=float(move.mean()) if len(move) else None)


def _days(config):
  """Official trading day per cursor, the initial balance and, under a rule
  gate, the rule zone in points per side (None without one)."""
  from gym_trading_env.research import ts_v1_corpus as C
  from gym_trading_env.research.fg_s3_quality_teacher_v2 import official_days
  env = C.native_env(config, market=False)
  zones = (None if env.rule_gate is None else
           {side: float(cfg['entry']['zone_points']) for side, cfg in env.rule_gate.items()})
  return official_days(env), float(env.config.trading.initial_balance), zones


def _random_job(job):
  """One random-entry account (or, kind 'always_add', the account that ADDs
  at every legal decision) over the first `steps` minutes."""
  from gym_trading_env.research import ts_v1_corpus as C
  env = C.native_env(job['config'], market=False)
  policy = C.Behavior(dict(kind=job.get('kind', 'probabilistic'), probability=job['probability'],
                           seed=[job['seed']]))
  days, initial, zones = _days(job['config'])
  obs, _ = env.reset(seed=0)
  rows, done = [], False
  while len(rows) < job['steps'] and not done:
    row = dict(cursor=int(env.current_step), legal=bool(obs['action_mask'][1] > 0.5),
               equity=float(env._calculate_equity()))
    row['executed'] = int(policy(env, obs))
    obs, reward, terminated, truncated, _ = env.step(row['executed'])
    row['reward'] = float(reward)
    rows.append(row)
    done = terminated or truncated
  out = metrics(rows, env.attributions, list(env.entries.values()), days, initial, env.fills, zones)
  return dict(seed=job['seed'], complete=bool(done and env.terminated), **out)


def exam(policy, out, label, config, source, steps=None, seeds=5, device=None, log=print, ts=None):
  """The student's Teacher-free run on the Native env of `config` (all of it,
  or its first `steps` minutes), `seeds` random-entry accounts and the
  always-ADD account on the same minutes; writes and returns out/exam.json."""
  from .run import _write, collect, load_policy, prune_collection
  from gym_trading_env.research import ts_v1_corpus as C
  from gym_trading_env.research.ts_v1_account import StudentEnv
  began, out = time.time(), Path(out)
  agent, meta = load_policy(policy, device, ts=ts)
  collect(agent, StudentEnv(C.native_env(config)), out / 'collect', steps or 10 ** 9, 'policy',
          1024, meta['policy'], log=log)
  rows = [json.loads(line) for f in sorted((out / 'collect' / 'steps').glob('*.jsonl'))
          for line in f.open()]
  with (out / 'collect' / 'state.pkl').open('rb') as f:
    run = pickle.load(f)['run']
  if any(r['mode'] != 'policy' or r['forced'] or r['explored'] for r in rows):
    raise ValueError(f'{out}: an exam is the formal policy only')
  days, initial, zones = _days(config)
  student = metrics([dict(r, legal=r['mask'][1] > 0.5) for r in rows], run['env']['attributions'],
                    list(run['env']['entries'].values()), days, initial, run['env']['fills'], zones)
  probability = student['adds'] / max(student['legal'], 1)
  jobs = [dict(config=str(config), probability=probability, seed=s, steps=len(rows))
          for s in range(seeds)]
  jobs.append(dict(config=str(config), kind='always_add', probability=1., seed=0, steps=len(rows)))
  # By module path, so spawned workers find the job also when this file runs as __main__.
  from dreamerv3.thinking_student.exam import _random_job as job
  with ProcessPoolExecutor(max(1, min(len(jobs), os.cpu_count())),
                           mp_context=multiprocessing.get_context('spawn')) as pool:
    *randoms, always = list(pool.map(job, jobs))
  summary = {k: dict(mean=float(np.mean([x[k] for x in randoms])),
                     std=float(np.std([x[k] for x in randoms])),
                     min=float(np.min([x[k] for x in randoms])),
                     max=float(np.max([x[k] for x in randoms])))
             for k in SCORES} if randoms else {}
  report = dict(
      label=label, policy=str(policy), policy_name=meta['policy'], source=source,
      teacher_free=True, trained=False, complete=bool(run['done'] and run['env']['terminated']),
      inference=meta['inference'], student=student,
      random=dict(probability=probability, seeds=[x['seed'] for x in randoms],
                  per_seed=randoms, summary=summary),
      always_add=always,
      advantage={k: student[k] - summary[k]['mean'] for k in SCORES if k in summary},
      advantage_per_seed=[student['reward_per_minute'] - x['reward_per_minute'] for x in randoms],
      advantage_always_add={k: student[k] - always[k] for k in SCORES},
      definitions=DEFINITIONS, seconds=round(time.time() - began, 1))
  # Last and whole: an exam.json on the disk is a finished exam (formal_train).
  _write(out / 'exam.json', json.dumps(report, indent=1).encode())
  log(f'{label} {meta["policy"]}: reward {student["reward"]:.3f} over {student["minutes"]} minutes, '
      f'random mean {summary.get("reward", {}).get("mean")}, always-ADD {always["reward"]:.3f}'
      + ('' if zones is None else f', rule compliance {student["rule_compliance"]}'))
  # Read from now on through exam.json, steps/ and state.pkl only (the paper
  # report, the review export): replay/ and the older snapshots go. A failure
  # here is logged and leaves the finished exam as it is.
  try:
    n, freed = prune_collection(out / 'collect', replay=True)
    log(f'pruned {n} paths of {out / "collect"}: {freed / 2 ** 20:.1f} MiB')
  except Exception as e:
    log(f'pruning {out / "collect"} FAILED {type(e).__name__}: {e}')
  return report


def paper_row(contract, pool):
  """The contract's row of the paper-trading pool: role 'paper' and never
  eligible for training, so neither a TRAIN nor a VALIDATION row can be
  paper-traded through it."""
  from gym_trading_env.research import ts_v1_corpus as C
  row = C.pool_row(contract, pool)
  if row.get('role') != 'paper' or row.get('eligible_for_training'):
    raise ValueError(f'{contract} is not a PAPER row of {pool} (role {row.get("role")})')
  return row


def paper_inputs(row, folder, rules, strategy=None):
  """write_inputs of a PAPER row with the TP / SL distances of `rules`
  (entry.take_profit_points / stop_loss_points in ticks; none: TP5 / SL30), or with
  a V2 `strategy` (load_strategy) as for the TRAIN corpora and DEV."""
  import yaml
  from gym_trading_env.research import ts_v1_corpus as C
  if strategy and rules:
    raise ValueError('a V2 strategy sets its own exit rule: no TP / SL distances with it')
  config = C.write_inputs(row, folder, strategy=strategy)
  if rules:
    env = yaml.safe_load(config.read_text())
    env['entry'].update(rules)
    config.write_text(yaml.safe_dump(env, sort_keys=False))
  return config


def strategy_record(path):
  """version and sha256 of a strategy file (None: no strategy)."""
  if not path:
    return None
  import hashlib
  import yaml
  data = Path(path).read_bytes()
  return dict(path=str(Path(path).resolve()), version=(yaml.safe_load(data) or {}).get('version'),
              sha256=hashlib.sha256(data).hexdigest())


def paper_strategy(policy, given=None):
  """(paper strategy, training strategy) of a PAPER exam of `policy` (user 2026-11-06:
  "我们要支持v2.4这种训练和paper有差异的方式"). The training strategy is paths.strategy of the
  policy's run (policies/pi_XXXX/../../config.yaml); the paper strategy is `given`
  (--strategy), else the run's paths.paper_strategy, else its training strategy."""
  import yaml
  run = Path(policy).resolve().parents[1] / 'config.yaml'
  paths = (yaml.safe_load(run.read_text()) or {}).get('paths', {}) if run.exists() else {}
  train = paths.get('strategy')
  return given or paths.get('paper_strategy') or train, train


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--policy', required=True, help='frozen policy folder')
  parser.add_argument('--out', required=True, help='exam folder')
  parser.add_argument('--pool', default=None, help='TS market pool with the VALIDATION row (DEV)')
  parser.add_argument('--contract', default=None, help='DEV contract of --pool, PAPER contract of --paper-pool')
  parser.add_argument('--paper-pool', default=None,
                      help='paper-trading pool (markets/paper_pool.json) with the row of --contract (PAPER)')
  parser.add_argument('--take-profit-points', type=int, default=None,
                      help='PAPER only: TP distance in ticks of the product (env default 5)')
  parser.add_argument('--stop-loss-points', type=int, default=None,
                      help='PAPER only: SL distance in ticks of the product (env default 30)')
  parser.add_argument('--strategy', default=None,
                      help='DEV and PAPER: V2 strategy overlay (gym ts_v1_corpus.load_strategy)')
  parser.add_argument('--beta-q', type=float, default=None,
                      help='DEV only: decision beta_Q of the frozen policy (E1 diagnostic, user 2026-10-30)')
  parser.add_argument('--corpus', default=None, help='TRAIN corpus folder (TRAIN_SMOKE)')
  parser.add_argument('--steps', type=int, default=None, help='first minutes only (default: all)')
  parser.add_argument('--random-seeds', type=int, default=5)
  parser.add_argument('--device', default=None, help="default: the snapshot's device")
  args = parser.parse_args(argv)
  if sum(map(bool, (args.pool, args.paper_pool, args.corpus))) != 1:
    parser.error('give --pool and --contract (DEV), --paper-pool and --contract (PAPER) '
                 'or --corpus (TRAIN_SMOKE)')
  rules = {k: v for k, v in (('take_profit_points', args.take_profit_points),
                              ('stop_loss_points', args.stop_loss_points)) if v is not None}
  if rules and not args.paper_pool:
    parser.error('--take-profit-points / --stop-loss-points are for PAPER only')
  if args.beta_q is not None and not args.pool:
    parser.error('--beta-q is for DEV only')
  if args.strategy and args.corpus:
    parser.error('--strategy is for DEV and PAPER; a TRAIN corpus has its own inputs')
  if args.strategy and rules:
    parser.error('--strategy sets the exit rule: no --take-profit-points / --stop-loss-points with it')
  ts = None if args.beta_q is None else {'decision.beta_q': args.beta_q}
  from gym_trading_env.research import ts_v1_corpus as C
  if args.pool:
    row = C.dev_row(args.contract, args.pool)
    config = C.write_inputs(row, Path(args.out) / 'inputs',
                            strategy=args.strategy and C.load_strategy(args.strategy))
    label, source = 'DEV', dict(pool=str(Path(args.pool).resolve()), row=row, ts_override=ts,
                                strategy=args.strategy and str(Path(args.strategy).resolve()))
  elif args.paper_pool:
    row = paper_row(args.contract, args.paper_pool)
    strategy, train = (args.strategy, None) if rules else paper_strategy(args.policy, args.strategy)
    config = paper_inputs(row, Path(args.out) / 'inputs', rules,
                          strategy and C.load_strategy(strategy))
    label, source = 'PAPER', dict(pool=str(Path(args.paper_pool).resolve()), row=row, rules=rules,
                                  strategy=strategy and str(Path(strategy).resolve()),
                                  paper_strategy=strategy_record(strategy), train_strategy=strategy_record(train))
  else:
    config = Path(args.corpus) / 'inputs' / 'env.yaml'
    meta = json.loads((Path(args.corpus) / 'source.json').read_text())
    label, source = 'TRAIN_SMOKE', dict(corpus=args.corpus, contract=meta['contract'],
                                        fold=meta['fold'])
  exam(args.policy, args.out, label, config, source, args.steps, args.random_seeds, args.device, ts=ts)


if __name__ == '__main__':
  main()
