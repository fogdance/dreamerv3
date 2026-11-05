"""Thinking Student V2.3 (user 2026-11-05: "规则决定学生能在哪里交易；大脑决定
在这些位置中什么时候交易"): the exam under a rule gate. Every account (the
student, the random ones and the always-ADD one) reports its entries by stop
source, its rule compliance and its fills inside the rule zone; an entry off
the rule positions of a gated env fails the exam; the paper report chains
these numbers and shows a dash for exams written before them.
Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from dreamerv3.thinking_student import exam as E
from dreamerv3.thinking_student import paper as P

CONFIGS = Path(E.__file__).parents[2] / 'configs/thinking_student'
POOL = CONFIGS / 'markets/pool.json'
# V2.4 (user 2026-11-05, "止损固定5点"): the rule gate, every stop fixed at the fill +/- 5.
STRATEGY = CONFIGS / 'markets/strategy_v2_4_fixed_stop.yaml'
ZONES = dict(SHORT=5., LONG=5.)
SHORT = dict(reason='ENTRY', side='SHORT', price='97', line='100', stop_source='RULE', rule_event='X@5', touch=1)
LONG = dict(reason='ENTRY', side='LONG', price='108', line='100', stop_source='CAPPED', rule_event='Y@9', touch=2)


def rows(n=4):
  return [dict(cursor=c, legal=True, executed=int(c % 2 == 0), reward=0., equity=100.) for c in range(n)]


def test_the_exam_counts_entries_by_stop_and_rule():
  days = {c: 'd1' for c in range(4)}
  fills = [SHORT, dict(reason='TAKE_PROFIT', side='SHORT', price='91'), LONG]
  m = E.metrics(rows(), [], [], days, 100., fills, ZONES)
  # The SHORT fill is 3 below its line (inside [H-5, H]); the LONG one 8 above (outside [L, L+5]).
  assert (m['filled_entries'], m['rule_entries'], m['zone_entries']) == (2, 2, 1)
  assert (m['rule_compliance'], m['fill_in_zone']) == (1., .5)
  assert m['stop_sources'] == dict(RULE=1, CAPPED=1)
  # Without a gate: the stop sources only.
  m = E.metrics(rows(), [], [], days, 100., fills)
  assert (m['rule_entries'], m['zone_entries'], m['rule_compliance'], m['fill_in_zone']) == (None,) * 4
  assert m['stop_sources'] == dict(RULE=1, CAPPED=1)
  assert E.metrics(rows(), [], [], days, 100.)['stop_sources'] == {}
  # A gated account with an entry off its rule positions can only be a bug.
  with pytest.raises(ValueError, match='1 entries off its rule positions'):
    E.metrics(rows(), [], [], days, 100., [SHORT, dict(LONG, rule_event=None)], ZONES)


def exam_folder(folder, contract, net, always=None, student=None):
  """The files of a PAPER exam that the paper report reads."""
  steps = folder / 'collect' / 'steps'
  steps.mkdir(parents=True)
  steps.joinpath('0.jsonl').write_text('\n'.join(json.dumps(dict(
      time=f'2025-0{1 + i}-02 09:0{i}:00', equity=100. + net * i, mask=[1., 1.], z=1., correction=1.))
      for i in (0, 1)) + '\n')
  s = dict(minutes=2, trading_days=1, adds=1, add_rate=.5, tp=1, sl=0, open_trades=0, occupancy=.5,
           reward_per_minute=0., net_pnl=net, max_drawdown=0., **(student or {}))
  report = dict(label='PAPER', teacher_free=True, trained=False, complete=True,
                source=dict(row=dict(contract=contract)), student=s,
                random=dict(per_seed=[dict(net_pnl=0., reward_per_minute=0.)]))
  if always is not None:
    report['always_add'] = dict(net_pnl=always)
  (folder / 'exam.json').write_text(json.dumps(report))


def test_the_paper_report_chains_the_rule_numbers_and_shows_a_dash_for_older_exams(tmp_path):
  rule = lambda n, inside, sources: dict(filled_entries=n, rule_entries=n, zone_entries=inside,
                                         rule_compliance=1., fill_in_zone=inside / n, stop_sources=sources)
  exam_folder(tmp_path / 'v23' / 'A', 'CZCE.A', 10., -5., rule(4, 3, dict(RULE=3, CAPPED=1)))
  exam_folder(tmp_path / 'v23' / 'B', 'CZCE.B', 20., -7., rule(6, 6, dict(RULE=5, GAP=1)))
  report = P.policy_report(tmp_path / 'v23', ['A', 'B'])
  whole = report['whole']
  assert (whole['net_pnl'], whole['always_net_pnl']) == (30., -12.)
  assert (whole['rule_compliance'], whole['fill_in_zone']) == (1., .9)
  assert whole['stop_sources'] == dict(RULE=8, CAPPED=1, GAP=1)
  text = P.markdown(dict(contracts=['CZCE.A', 'CZCE.B'], policies={'pi': dict(report, rules=dict(strategy='s.yaml'))}))
  assert '| -12 |' in text and '| 1.000 | 0.900 | CAPPED 1, GAP 1, RULE 8 |' in text
  # V2.2 exams: no always-ADD account, no rule numbers.
  exam_folder(tmp_path / 'v22' / 'A', 'CZCE.A', 10.)
  old = P.policy_report(tmp_path / 'v22', ['A'])
  assert [old['whole'][k] for k in ('always_net_pnl', 'rule_compliance', 'fill_in_zone', 'stop_sources')] == [None] * 4
  text = P.markdown(dict(contracts=['CZCE.A'], policies={'pi': dict(old, rules=dict(strategy='s.yaml'))}))
  assert '| 1/1 | — |' in text and text.count('| — | — | — |') == 1


@pytest.mark.skipif(not (POOL.exists() and STRATEGY.exists()), reason='Thinking Student V2.4 configs missing')
def test_under_the_rule_gate_both_baselines_enter_at_rule_positions_only(tmp_path):
  from gym_trading_env.research import ts_v1_corpus as C
  config = C.write_inputs(C.train_row('CZCE.FG205', POOL), tmp_path / 'in', strategy=C.load_strategy(STRATEGY))
  opens = np.flatnonzero(C.native_env(config, market=False)._gate('SHORT')['open'])
  steps = int(opens[min(60, len(opens) - 1)]) + 1  # the first 60 rule positions
  always = E._random_job(dict(config=str(config), kind='always_add', probability=1., seed=0, steps=steps))
  rnd = E._random_job(dict(config=str(config), probability=.3, seed=1, steps=steps))
  assert always['adds'] == always['legal'] > 0  # an ADD at every legal decision
  for x in (always, rnd):
    assert x['minutes'] == steps and x['filled_entries'] > 0
    assert x['rule_entries'] == x['filled_entries'] and x['rule_compliance'] == 1.
    assert set(x['stop_sources']) == {'FIXED'} and 0 < x['fill_in_zone'] <= 1
