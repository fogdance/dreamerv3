import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def offline_pretrain(
    make_agent,
    make_replay,
    make_stream,
    make_logger,
    args,
    replay_dir: str | None = None,
):
  """
  纯离线预训练：
  - 不创建 env；
  - 不调用 replay.add(...)；
  - 只从已有 replay 中 sample batch，反复调用 agent.train(...)。

  约定：
  - logger.step（即 step）直接表示“已经做了多少次梯度更新”；
  - config.run.steps 就是“希望做多少个 gradient steps”。
  """

  # -----------------------------
  # 1. 基础组件：agent / replay / logger
  # -----------------------------
  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step            # 现在用它表示 'num_gradient_updates'
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  report_agg = elements.Agg()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length

  # 只保留 log / report / save 的节奏控制
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  # -----------------------------
  # 2. 加载现有 replay（模型 A 的数据）
  # -----------------------------
  if replay_dir is not None:
    print(f'[offline_pretrain] Loading replay from: {replay_dir}')
    replay.load(directory=replay_dir)
  else:
    # 使用 replay 自己的 directory（通常是 logdir/replay）
    print('[offline_pretrain] Loading replay from default directory')
    replay.load()

  print(
      f'[offline_pretrain] replay_dir={replay_dir or replay.directory} '
      f'items={len(replay.items)} sampler={len(replay.sampler)} '
      f'chunks={len(replay.chunks)} length={replay.length}'
  )

  stats = replay.stats()
  print(
      f'[offline_pretrain] loaded: items={len(replay)} '
      f'chunks={len(replay.chunks)} ram_gb={stats["ram_gb"]:.2f}'
  )

  # 没有足够数据就直接退出（生产环境最好直接抛错）
  min_items = args.batch_size * args.batch_length
  if len(replay) < min_items:
    raise RuntimeError(
        f'[offline_pretrain] Replay too small: len(replay)={len(replay)} '
        f'< batch_size * batch_length = {min_items}'
    )

  # -----------------------------
  # 3. 从 replay 构造训练 / report 流
  # -----------------------------
  # make_stream(replay, mode) -> Stateless/Consec 流
  # agent.stream(...) -> 把 JAX 需要的前后缀（replay_context 等）打好
  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]  # list 是为了闭包内可变
  carry_report = agent.init_report(args.batch_size)

  # -----------------------------
  # 4. Checkpoint：支持断点续训
  # -----------------------------
  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  # 如果你希望恢复 sampler/priorities 状态，也可以把 replay 加进去：
  cp.replay = replay

  if getattr(args, 'from_checkpoint', None):
    regex = getattr(args, 'from_checkpoint_regex', None)
    print(f'[offline_pretrain] Loading checkpoint from: {args.from_checkpoint}')
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=regex),
    ))

  # 如果 ckpt 目录已有最新 checkpoint，会自动 load；否则会保存初始状态
  cp.load_or_save()

  print('Start OFFLINE pretrain loop')
  print(
      f'[offline_pretrain] target gradient steps = {int(args.steps)} '
      f'(logger.step will count gradient updates directly)'
  )

  # -----------------------------
  # 5. 纯离线训练主循环（无 env，无 driver）
  # -----------------------------
  # 现在的逻辑：
  #   while step < args.steps:
  #       batch = next(stream_train)
  #       agent.train(...)
  #       step.increment()
  #
  # 所有数据都来自之前加载好的 replay。
  # -----------------------------
  while step < args.steps:

    # 1 个 batch = 1 次 gradient update
    with elements.timer.section('offline_stream_next'):
      batch = next(stream_train)

    carry_train[0], outs, mets = agent.train(carry_train[0], batch)
    train_fps.step(batch_steps)

    # 如果 agent 产生了 priority update，则更新 replay.sampler
    if 'replay' in outs:
      replay.update(outs['replay'])

    train_agg.add(mets, prefix='train')
    step.increment()  # 这里 +1 就是 “多做了一次梯度更新”

    # 报表（可选）：在离线 pretrain 阶段也可以看 open-loop 等指标
    if should_report(step):
      report_agg.reset()
      for _ in range(args.consec_report * args.report_batches):
        with elements.timer.section('offline_stream_report'):
          batch_rep = next(stream_report)
        carry_report, mets = agent.report(carry_report, batch_rep)
        report_agg.add(mets)
      logger.add(report_agg.result(), prefix='report')

    # 写日志
    if should_log(step):
      print(
          f'\n[offline_pretrain] global_step={int(step)} / target={int(args.steps)}',
          flush=True,
      )
      logger.add(train_agg.result())
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    # 保存 checkpoint
    if should_save(step):
      cp.save()

  logger.close()
  print('OFFLINE pretrain done.')
