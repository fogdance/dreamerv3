"""Thinking Student V2 minimal change (V0.5, user 2026-10-31): the student
without MACD and with 48 complete 15m buckets, the F1 inheritance into its
longer context, the R2 return in the net PnL reward's unit and the formal
settings that carry them.
Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import jax
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.thinking_student import formal_train as FT
from dreamerv3.thinking_student import labels as L
from dreamerv3.thinking_student import q_return as QR
from dreamerv3.thinking_student import transfer
from dreamerv3.thinking_student.config import observation_space
from dreamerv3.thinking_student.student import make_agent

V2 = {'agent.ts.market.macd': False, 'agent.ts.market.context_15m': 48, 'agent.ts.outcome_points': 10.}


def student(folder, updates=None):
  return make_agent(folder, preset='small', batch_size=1, batch_length=1, updates=updates)


def observation(market, seed=0):
  """One decision's inputs of the shapes the student reads, valid market rows."""
  rng = np.random.default_rng(seed)
  obs = {}
  for k, space in observation_space(market).items():
    obs[k] = (rng.normal(0, .1, (1, *space.shape)) if space.dtype != bool else np.zeros((1, *space.shape))
              ).astype(space.dtype)
  obs['market_context_15m'][..., 6] = 1
  obs['market_macd_5m'][..., 3] = 1
  obs['action_mask'][:] = 1
  obs['is_first'][:], obs['is_last'][:], obs['is_terminal'][:] = True, False, False
  return obs


def tokens(agent, obs):
  """The market encoder's output (the market brain's input) of one decision."""
  fn = jax.jit(nj.pure(lambda o: agent.model.market_enc(o, o['is_first'], False, single=True)),
               static_argnames=('create', 'modify'))
  _, out = fn(agent.params, obs, seed=0, create=False, modify=False)
  return np.asarray(out, np.float32)


def test_the_v2_student_reads_48_buckets_and_no_macd(tmp_path):
  agent = student(tmp_path / 'v2', V2)
  assert dict(agent.config.ts.market) == {'macd': False, 'context_15m': 48}
  assert agent.config.ts.outcome_points == 10.
  assert not any('macd5' in k for k in agent.params)
  assert agent.params['market_enc/context15/kernel'].shape[0] == 48 * 7
  obs = observation(dict(macd=False, context_15m=48))
  base = tokens(agent, obs)
  # MACD reaches no parameter: any value of it leaves the market input as it was.
  macd = obs['market_macd_5m'].copy()
  macd[..., :3] += 3.
  np.testing.assert_array_equal(tokens(agent, dict(obs, market_macd_5m=macd)), base)
  # The 24 older buckets are inputs.
  older = obs['market_context_15m'].copy()
  older[:, :24, :4] += 1.
  assert np.abs(tokens(agent, dict(obs, market_context_15m=older)) - base).max() > 1e-3
  # A 24-bucket context is refused, never reshaped into the 48-bucket input.
  with pytest.raises(Exception, match='15m context'):
    tokens(agent, dict(obs, market_context_15m=obs['market_context_15m'][:, 24:]))


def test_the_v1_student_is_unchanged(tmp_path):
  agent = student(tmp_path / 'v1')
  assert dict(agent.config.ts.market) == {'macd': True, 'context_15m': 24}
  assert agent.config.ts.outcome_points == L.MAE_SCALE == 30.
  assert agent.params['market_enc/context15/kernel'].shape[0] == 24 * 7
  obs = observation(dict(macd=True, context_15m=24))
  macd = obs['market_macd_5m'].copy()
  macd[..., :3] += 3.
  assert np.abs(tokens(agent, dict(obs, market_macd_5m=macd)) - tokens(agent, obs)).max() > 1e-3


def test_f1_context_rows_go_to_the_newest_buckets():
  source = {'enc/context15/kernel': np.arange(24 * 7 * 2, dtype=np.float32).reshape(24 * 7, 2),
            'enc/fusion/kernel': np.ones((6, 3), np.float32)}
  target = {'market_enc/context15/kernel': np.zeros((48 * 7, 2), np.float32),
            'market_enc/fusion/kernel': np.zeros((6, 3), np.float32), 'decision/w': np.zeros(1)}
  values, reused, _, widened, initialized = transfer.plan(source, target)
  kernel = values['market_enc/context15/kernel']
  np.testing.assert_array_equal(kernel[:24 * 7], 0)
  np.testing.assert_array_equal(kernel[24 * 7:], source['enc/context15/kernel'])
  assert widened == ['market_enc/context15/kernel'] and reused == ['market_enc/fusion/kernel']
  assert initialized == ['decision/w']


def test_the_net_pnl_return_is_in_the_reward_unit():
  class Clock:  # one market minute per row
    prefix = np.arange(12.)
  trades = [dict(entry_cursor=2, exit_cursor=4, net_trade_pnl=-618.),
            dict(entry_cursor=5, exit_cursor=10, net_trade_pnl=1182.)]
  g = QR.window_returns(Clock, 0, 10, trades, horizon=3, scale=618.)
  np.testing.assert_allclose(g[:2], -1.)
  np.testing.assert_allclose(g[4:], 1182. / 618.)


def test_the_formal_settings_carry_the_market_and_the_strategy():
  cfg = FT.load_config(FT.Path(FT.__file__).parents[2] / 'configs/thinking_student/trial_v2_minimal.yaml')
  argv = [str(a) for a in FT._student(cfg)]
  assert argv[argv.index('--market-macd') + 1] == '0' and argv[argv.index('--context-15m') + 1] == '48'
  assert float(argv[argv.index('--outcome-points') + 1]) == 10.
  assert FT._strategy(cfg)[1].endswith('strategy_v2_short_structural.yaml')
  assert cfg['iteration']['q_return'] == QR.NET_PNL
  assert cfg['foundation']['save_every'] == cfg['iteration']['save_every'] == 0
  v1 = FT.load_config(FT.Path(FT.__file__).parents[2] / 'configs/thinking_student/formal_v1_graded.yaml')
  argv = [str(a) for a in FT._student(v1)]
  assert argv[argv.index('--market-macd') + 1] == '1' and argv[argv.index('--context-15m') + 1] == '24'
  assert float(argv[argv.index('--outcome-points') + 1]) == 30.
  assert FT._strategy(v1) == []


def test_mae_and_mfe_labels_are_in_units_of_the_planned_risk():
  v2 = {'inputs': {'strategy': {'entry': {'stop_rule': 'prior_15m_high', 'max_risk_points': 10}}}}
  assert L.outcome_points(v2) == 10. and L.outcome_points({'inputs': {'strategy': None}}) == 30.
  assert L.outcome_points({'inputs': {}}) == 30.  # a V1 corpus built before the key
  n = 2
  facts = {k: np.zeros(n, bool) for k in ('filled', 'early10_5', 'early10_5_valid', 'early10_15',
                                         'early10_15_valid', 'mae_15_complete', 'mae_60_complete',
                                         'path_complete')}
  facts.update(filled=np.array([True, False]), kind=np.zeros(n, np.int8), held_minutes=np.full(n, 30),
               mae_15=np.array([5., 0.]), mae_60=np.zeros(n), mae_full=np.zeros(n), reward=np.zeros(n),
               mae_15_complete=np.array([True, False]))
  graded = {f'mfe_{h}': np.array([20., 0.]) for h in (15, 60, 120)}
  graded.update({f'mfe_{h}_complete': np.array([True, False]) for h in (15, 60, 120)})
  out = L.outcome_labels(facts, ('TP', 'SL', 'OTHER', 'OPEN', 'CENSORED'), graded, 10.)
  assert out['outcome_mae'][0, 0] == .5 and out['outcome_mfe'][0, 0] == 2.
  assert L.outcome_labels(facts, ('TP', 'SL', 'OTHER', 'OPEN', 'CENSORED'), graded)['outcome_mae'][0, 0] == np.float32(5 / 30)


def test_paper_trading_takes_the_v2_strategy_in_place_of_tp_sl_distances(tmp_path):
  """User 2026-11-04: paper trading of a V2 policy on FG501-FG605 under the
  V2 strategy of its corpora; fixed TP / SL distances and beta_Q are refused
  with it, and a TRAIN corpus takes no overlay."""
  from gym_trading_env.research import ts_v1_corpus as C
  from gym_trading_env.rewards.single3_reward import NET_PNL_SCHEMA
  from dreamerv3.thinking_student import exam as E
  configs = FT.Path(FT.__file__).parents[2] / 'configs/thinking_student'
  strategy, pool = configs / 'markets/strategy_v2_short_structural.yaml', configs / 'markets/paper_pool.json'
  base = ['--policy', str(tmp_path), '--out', str(tmp_path / 'x'), '--contract', 'CZCE.FG501']
  for extra in (['--paper-pool', str(pool), '--strategy', str(strategy), '--stop-loss-points', '5'],
                ['--paper-pool', str(pool), '--beta-q', '0'],
                ['--corpus', str(tmp_path), '--strategy', str(strategy)]):
    with pytest.raises(SystemExit):
      E.main(base + extra)
  assert not (tmp_path / 'x').exists()
  with pytest.raises(ValueError, match='own exit rule'):
    E.paper_inputs(E.paper_row('CZCE.FG501', pool), tmp_path / 'y', {'stop_loss_points': 5},
                   C.load_strategy(strategy))
  config = E.paper_inputs(E.paper_row('CZCE.FG501', pool), tmp_path / 'in', {}, C.load_strategy(strategy))
  env = C.native_env(config, market=False)
  assert env.stop_rule['max_risk_points'] == 10 and env.context_15m_buckets == 48
  assert env.reward_config.schema == NET_PNL_SCHEMA and float(env.reward_config.reward_scale_cny) == 618.


def test_the_paper_report_names_the_strategy_not_the_v1_distances():
  """A V2 paper run trades each trade's own SL / TP: its report names the
  strategy file, never the V1 default 5 / 30."""
  from dreamerv3.thinking_student import paper as P
  whole = dict(first='2024-08-19', last='2025-12-31', net_pnl=-1., max_drawdown=1., positive_months=0,
               months=1, adds=1, tp=0, sl=1, monthly={'2024-08': -1.},
               random_net_pnl=dict(mean=0., std=0., beaten=0, seeds=5))

  def report(rules):
    return P.markdown(dict(contracts=['CZCE.FG501'], policies={'pi': dict(whole=whole, contracts=[], rules=rules)}))
  v2 = report(dict(strategy='strategy_v2_short_structural.yaml'))
  assert '| pi | strategy strategy_v2_short_structural.yaml |' in v2 and '5 / 30' not in v2
  assert '| pi | 5 / 30 |' in report(dict(take_profit_points=5, stop_loss_points=30))


POOL = FT.Path(FT.__file__).parents[2] / 'configs/thinking_student/markets/pool.json'


@pytest.mark.skipif(not POOL.exists(), reason='Thinking Student pool not present')
def test_check_collection_measures_a_long_trade_on_its_own_side(tmp_path):
  """V2.1 (user 2026-11-04): a LONG's TP lies above its fill and its SL below;
  both distances are positive, so a LONG TP short of its target or an SL
  above its stop is a violation, as it is for a SHORT."""
  import json
  import pickle
  from decimal import Decimal
  from gym_trading_env.research import ts_v1_corpus as C
  from dreamerv3.thinking_student import run as R
  strategy = C.load_strategy(POOL.parent / 'strategy_v2_1_structural.yaml')
  inputs = C.write_inputs(C.train_row('CZCE.FG401', POOL), tmp_path / 'in', strategy=strategy)
  env = C.native_env(inputs, market=False)
  t = 400
  fill = Decimal(str(env._price(env._raw.Open.iloc[t + 1], 'LONG', env.bobs[t + 1], opening=True)))
  plan = env.planned_exit(t, fill, 'LONG')
  tp, sl = Decimal(plan['take_profit_price']), Decimal(plan['stop_loss_price'])
  assert sl < fill < tp
  row = dict(step=0, cursor=0, time='x', mode='policy', mask=[1., 1.], proposed=0, executed=0, forced=False,
             mu_prob=1., explored=False, z=-1., reward=0., **{k: 0 for k in R.ACCOUNT_KEYS})

  def violations(name, exit_price, reason):
    out = tmp_path / name
    (out / 'steps').mkdir(parents=True)
    (out / 'steps' / '000000.jsonl').write_text(json.dumps(row) + '\n')
    trade = dict(entry_cursor=t + 1, entry_price=str(fill), exit_price=str(exit_price), exit_cursor=t + 9,
                 reason=reason, side='LONG')
    with (out / 'state.pkl').open('wb') as f:
      pickle.dump({'run': {'env': {'attributions': [trade], 'entries': {}}}}, f)
    return R.check_collection(out, inputs=inputs)['point_violations']
  assert not violations('tp', tp, 'TAKE_PROFIT') and not violations('sl', sl, 'STOP_LOSS')
  assert violations('tp_short', tp - 1, 'TAKE_PROFIT') and violations('sl_short', sl + 1, 'STOP_LOSS')
