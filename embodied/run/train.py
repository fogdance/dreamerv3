import collections
from functools import partial as bind

import elements
import embodied
import numpy as np

from .action_mask_warmup import ActionMaskWarmup, METRICS, ADD_RECALL_METRICS
from .checkpoint_retention import StepCheckpointRetention


def _policy_with_logging(agent, *args):
  carry, acts, outs = agent.policy(*args, mode='train')
  if 'p_add' in outs:
    # Driver logs diagnostics; Replay excludes the established log/ channel.
    outs = dict(outs)
    outs['log/p_add'] = outs.pop('p_add')
  return carry, acts, outs


def _apply_action_mask_warmup_gate(agent, warmup, step):
  if not warmup.enabled:
    return
  if not hasattr(agent, 'set_avail_actor_enabled'):
    raise TypeError("Auto action-mask warm-up requires a compatible agent")
  if hasattr(agent, 'get_avail_actor_enabled') and agent.get_avail_actor_enabled():
    warmup.actor_enabled = True
    if warmup.switched_step < 0:
      warmup.switched_step = int(step)
  agent.set_avail_actor_enabled(warmup.actor_enabled)


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  warmup_agg = elements.Agg()
  warmup_samples = [0]
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  transfer = args.get('fg_stage_transfer', '')
  receipt = {}
  if transfer:
    import json
    from pathlib import Path
    receipt = json.loads(Path(transfer).read_text())
  single3_transfer = receipt.get('source_kind') == 'single3_offline'
  gate_factory = ActionMaskWarmup.for_single3_1m if single3_transfer else ActionMaskWarmup
  warmup = gate_factory(args.action_mask_warmup, args.action_mask_actor_initial)

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  retention = StepCheckpointRetention(
      logdir, args.get('checkpoint_retention', {}),
      metadata=args.get('seed_protocol', {}))

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

  full_checkpoint = args.get('fg_full_checkpoint', False)
  if full_checkpoint and (args.envs != 1 or not args.get('env_checkpoint', False)):
    raise ValueError('FG full-state training requires one persistent environment')
  if not full_checkpoint:
    replay.load()
  print(f'[replay] dir={replay.directory} items={len(replay.items)} sampler={len(replay.sampler)} chunks={len(replay.chunks)} length={replay.length}')


  stream_train = agent.stream(make_stream(replay, 'train'))
  stream_report = agent.stream(make_stream(replay, 'report'))
  if not full_checkpoint:
    stream_train, stream_report = iter(stream_train), iter(stream_report)

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = [agent.init_report(args.batch_size)]
  runtime = None
  if full_checkpoint:
    from .fg_checkpoint import FullCheckpoint, RuntimeState, StepSchedule
    schedule = StepSchedule(args.fg_report_every_steps, args.fg_save_every_steps, args.fg_key_steps)
    def get_carry():
      return dict(train=carry_train[0], report=carry_report[0],
          episodes={k: v.reducers for k, v in episodes.items()})
    def set_carry(state):
      restore = getattr(agent, 'restore_carry', lambda x: x)
      carry_train[0], carry_report[0] = restore(state['train']), restore(state['report'])
      episodes.clear()
      for worker, reducers in state['episodes'].items():
        episodes[worker].reducers = reducers
    runtime = RuntimeState(schedule, should_train, (stream_train, stream_report),
        get_carry, set_carry, dict(train=train_agg, warmup=warmup_agg, epstats=epstats), warmup_samples)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    if runtime:
      runtime.start()
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
      # ADD counts need received metrics; JAX first returns an empty dict.
      # Preserve legacy sample accounting when the ADD check is disabled.
      if warmup.enabled and (mets or not warmup.add_recall_enabled):
        warmup_agg.add({
            key: mets[key]
            for key, _ in METRICS.values() if key in mets})
        if warmup.add_recall_enabled:
          # Sum confusion counts, not batch recalls (often 0 positives/batch).
          for prefix in ('post', 'prior'):
            for suffix in ('add_positives', 'add_true_positives'):
              key = f'avail/{prefix}_{suffix}'
              warmup_agg.add(key, mets[key], agg='sum')
        warmup_samples[0] += batch_steps
  driver.on_step(trainfn)

  cp = (FullCheckpoint(logdir / 'ckpt', step, args.fg_checkpoint_identity,
      (stream_train, stream_report), args.fg_key_steps, args.fg_reserve_gib)
      if full_checkpoint else elements.Checkpoint(logdir / 'ckpt', step=step))
  if full_checkpoint:
    cp.runtime = runtime
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  cp.action_mask_warmup = warmup
  persistent_env = args.get('env_checkpoint', False)
  if persistent_env:
    driver.reset(agent.init_policy)
    cp.driver = driver
  if args.from_checkpoint:
    regex = args.get('from_checkpoint_regex', None)
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=regex)))
  def save_checkpoint(final=False):
    if full_checkpoint:
      cp.save(final=final)
    else:
      cp.save()
      retention.maybe_retain(cp, step)

  if transfer and not cp.exists():
    if len(replay):
      raise ValueError('C recent stage requires empty Replay')
    components = dict(agent=bind(agent.load_exact, transfer=True) if full_checkpoint else agent.load)
    if not single3_transfer:
      components['action_mask_warmup'] = warmup.load
    elements.checkpoint.load(receipt['checkpoint'], components)
    if single3_transfer:
      # Begin the online observation window anew and publish pretrained weights.
      if agent.get_avail_actor_enabled():
        raise ValueError('Single3 offline source has an activated Actor/Critic Gate')
      agent.sync_policy_from_train(reset_counter=True)
    if full_checkpoint:
      # Initial recurrent states may depend on loaded parameters.
      carry_train[0] = agent.init_train(args.batch_size)
      carry_report[0] = agent.init_report(args.batch_size)
      driver.reset(agent.init_policy)
    print('C stage transfer: complete Agent/optimizer/gate; fresh recent Replay/accounts')

  cp.load_or_save()
  if not full_checkpoint:
    retention.maybe_retain(cp, step)
  if runtime and getattr(runtime, 'resume_started', False):
    runtime.start()
  if warmup.enabled:
    _apply_action_mask_warmup_gate(agent, warmup, step)
    phase = 'formal' if warmup.actor_enabled else 'warm-up'
    print(f'Action-mask training phase: {phase}')

  if args.get('fg_learning_audit', False):
    from .fg_learning_audit import snapshot_initial
    snapshot_initial(agent, str(logdir))

  print('Start training loop')
  policy = bind(_policy_with_logging, agent)
  if not persistent_env:
    driver.reset(agent.init_policy)
  while step < args.steps:

    count = min(10, schedule.distance(step, args.steps)) if full_checkpoint else 10
    driver(policy, steps=count)
    if full_checkpoint:
      report_due, save_due = schedule.advance(step)
    else:
      report_due, save_due = should_report(step), should_save(step)

    if report_due and len(replay):
      if runtime:
        runtime.start()
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report[0], mets = agent.report(carry_report[0], next(stream_report))
        agg.add(mets)
      report = agg.result()
      logger.add(report, prefix='report')
      switched = False
      gate_ready = warmup.ready(warmup_samples[0])
      if gate_ready:
        gate_metrics = warmup_agg.result()
        warmup_samples[0] = 0
        if warmup.add_recall_enabled:
          for prefix in ('post', 'prior'):
            positives = float(gate_metrics.get(f'avail/{prefix}_add_positives', 0))
            true_positives = float(gate_metrics.get(f'avail/{prefix}_add_true_positives', 0))
            gate_metrics[f'avail/{prefix}_add_recall'] = (
                true_positives / positives if positives else float('nan'))
        allowed_metrics = {**METRICS, **(ADD_RECALL_METRICS if warmup.add_recall_enabled else {})}
        gate_metrics = {
            key: gate_metrics[key]
            for key, _ in allowed_metrics.values() if key in gate_metrics} | {
            key: value for key, value in gate_metrics.items()
            if key.endswith('_add_positives') or key.endswith('_add_true_positives')}
        logger.add(gate_metrics, prefix='action_mask_gate_eval')
        switched = warmup.update(gate_metrics, step)
      warmup_metrics = warmup.metrics()
      warmup_metrics['pending_train_samples'] = warmup_samples[0]
      warmup_metrics['evaluation_ready'] = float(gate_ready)
      logger.add(warmup_metrics, prefix='action_mask_warmup')
      if switched:
        agent.set_avail_actor_enabled(True)
        print(f'Action-mask warm-up passed; masked actor enabled at step {int(step)}')
        if not full_checkpoint:
          save_checkpoint()

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if save_due:
      save_checkpoint(final=full_checkpoint and int(step) == int(args.steps))

  if persistent_env:
    logger.add(train_agg.result())
  if persistent_env or retention.has_due(step):
    save_checkpoint(final=full_checkpoint)

  if args.get('fg_learning_audit', False):
    from .fg_learning_audit import summarize_learning
    summarize_learning(agent, str(logdir), int(step))

  logger.close()
