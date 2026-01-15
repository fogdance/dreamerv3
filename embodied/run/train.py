import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    # try:
    #   # 只打印 worker==0，且每 200 步或回合结束时打印一次，避免刷屏
    #   if worker == 0 and ((int(step) % 200) == 0 or bool(tran.get('is_last', False))):
    #     elements.printing.print_("\n[DEBUG] Dump of tran keys/values:", flush=True)
    #     for k, v in tran.items():
    #       # 兼容 numpy/jax 标量和数组
    #       if hasattr(v, 'dtype') and hasattr(v, 'shape'):
    #         if getattr(v, 'ndim', None) == 0:
    #           # 标量：直接打印数值
    #           try:
    #             elements.printing.print_(f"  - {k}: scalar {float(v)} (dtype={v.dtype})")
    #           except Exception:
    #             elements.printing.print_(f"  - {k}: scalar {v} (dtype={v.dtype})")
    #         else:
    #           # 向量/图片/视频：只打印形状与 dtype，避免巨量输出
    #           elements.printing.print_(f"  - {k}: array shape={v.shape} dtype={v.dtype}")
    #       else:
    #         # Python 原生类型（bool/int/float/str 等）
    #         elements.printing.print_(f"  - {k}: {type(v).__name__} = {v}")
    #     elements.printing.print_("--------------------------------------\n", flush=True)
    # except Exception as e:
    #   elements.printing.print_(f"[DEBUG] tran dump failed: {e}")
  
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
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

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  replay.load()
  print(f'[replay] dir={replay.directory} items={len(replay.items)} sampler={len(replay.sampler)} chunks={len(replay.chunks)} length={replay.length}')


  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    regex = args.get('from_checkpoint_regex', None)
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
