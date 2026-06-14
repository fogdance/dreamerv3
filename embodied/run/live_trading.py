from collections import defaultdict
from functools import partial as bind
from datetime import datetime, time

import elements
import embodied
import numpy as np

from .action_mask_guard import require_formal_actor


def live_trading(make_agent, make_env, make_logger, args):
  """
  实盘 / 仿真盘 Runner：

  - 每次 driver(policy, steps=1)，只推进 1 根 K 线
  - env.step() live_mode 下会在 wait_kline_block() 里阻塞等新 K 线
  - 一旦 episode 结束（terminated 或 truncated），就退出主循环
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

  # from-env 的 episode 结束信号
  done_flag = {
      "value": False,
      "info": {},   # is_terminal / is_truncated 等附加信息
  }

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

    # ============ 关键：把 done 信号传回主循环 =============
    is_last = bool(np.asarray(tran.get('is_last', False)))
    is_terminal = bool(np.asarray(tran.get('is_terminal', False)))
    # 有些 wrapper 会额外塞 is_truncated，你这里 env._get_info 已经有 log/env/is_truncated
    is_truncated = bool(np.asarray(tran.get('is_truncated', False))) if 'is_truncated' in tran else (is_last and not is_terminal)

    if is_last:
      done_flag['value'] = True
      done_flag['info'] = dict(
          is_terminal=is_terminal,
          is_truncated=is_truncated,
      )

  # ---- Live 模式：强制单 env ----
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
  require_formal_actor(agent, "live_trading")

  print('Start LIVE trading loop (Ctrl+C to stop)')
  policy = lambda *p: agent.policy(*p, mode='eval')

  # 初始化 env + policy 状态
  driver.reset(agent.init_policy)

  cutoff = time(15, 0)  # 15:00
  cuton = time(20, 55)  # 20:55

  try:
    while True:
      now = datetime.now().time()
      if cutoff <= now < cuton:
        print('Reached cutoff time 15:00, exit LIVE loop.')
        break

      # 每次只走一步（当步如果没新 K，env 内部会阻塞到有新 K）
      driver(policy, steps=1)

      # 如果这一步 episode 结束（爆仓 / 风控 / end_idx / session_end），就退出 live loop
      if done_flag['value']:
        info = done_flag.get('info', {})
        print(
          "[live_trading] Episode finished, exit LIVE loop: "
          f"is_terminal={info.get('is_terminal')} "
          f"is_truncated={info.get('is_truncated')}"
        )
        break

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
