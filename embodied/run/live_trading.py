from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np
from datetime import datetime, time


def live_trading(make_agent, make_env, make_logger, args):
  """
  专门用于实盘 / 仿真盘的 Live Runner：

  - 使用你的 gym_trading_env（内部已经把 JuejinBarSource + wait_kline_block 封装好）
  - 每次 driver(policy, steps=1)，只推进 1 根 K 线
  - env.step() 里如果是 live_mode，会在 bar_source.wait_kline_block() 里阻塞到有新 K 线
  - 无限循环，直到 Ctrl+C
  """
  assert args.from_checkpoint, "live_trading 必须从 checkpoint 启动"

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    # 复用 episode 统计
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  # ---- Live 模式：一般只跑 1 个 env（1 个账户 / 1 个合约） ----
  if args.envs != 1:
    print(f'[live_trading] WARNING: live 模式强制使用单 env，忽略 args.envs={args.envs}')
  fns = [bind(make_env, 0)]

  driver = embodied.Driver(fns, parallel=False)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)

  # ---- 加载 checkpoint ----
  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])

  print('Start LIVE trading loop (Ctrl+C to stop)')
  policy = lambda *p: agent.policy(*p, mode='eval')

  # 让 driver 初始化 env 和 agent.policy 的隐状态
  driver.reset(agent.init_policy)

  cutoff = time(15, 0)  # 15:00
  cuton = time(20, 55)  # 20:55

  try:
    while True:
      if datetime.now().time() >= cutoff and datetime.now().time() < cuton:
        print('Reached cutoff time 15:00, exit LIVE loop.')
        break

      # === 核心区别：每轮只推进 1 个 env step ===
      #
      # 对于 live futures：
      #   - env.step() 内部会调用 JuejinBarSource.wait_kline_block()
      #   - wait_kline_block() 在 DB 没有新 K 线之前会阻塞
      #   - 一旦有新 K 线，该方法 reload window + 重建 MarketStore
      #   - 然后 env.step() 用最新的 df_market/store 执行一次仿真 + 返回 obs
      #
      driver(policy, steps=1)

      if should_log(step):
        logger.add(agg.result())
        logger.add(epstats.result(), prefix='epstats')
        logger.add(usage.stats(), prefix='usage')
        logger.add({'fps/policy': policy_fps.result()})
        logger.add({'timer': elements.timer.stats()['summary']})
        logger.write()
  except KeyboardInterrupt:
    print('LIVE loop interrupted by user.')
  finally:
    logger.close()
