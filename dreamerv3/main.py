import importlib
import os
import pathlib
import sys
from datetime import datetime
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml
import gym_trading_env


class LiveTerminalOutput(elements.logger.TerminalOutput):

  def __call__(self, summaries):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    step = max(s for s, _, _, in summaries)
    scalars = {
        k: float(v) for _, k, v in summaries
        if isinstance(v, np.ndarray) and len(v.shape) == 0}
    if self._pattern:
      scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    else:
      truncated = 0
      if len(scalars) > self._limit:
        truncated = len(scalars) - self._limit
        scalars = dict(list(scalars.items())[:self._limit])
    formatted = {k: self._format_value(v) for k, v in scalars.items()}
    if self._name:
      header = f'{"-" * 20}[{self._name} Step {step:_} {now}]{"-" * 20}'
    else:
      header = f'{"-" * 20}[Step {step:_} {now}]{"-" * 20}'
    content = ''
    if self._pattern:
      content += f"Metrics filtered by: '{self._pattern.pattern}'"
    elif 'truncated' in locals() and truncated:
      content += f'{truncated} metrics truncated, filter to see specific keys.'
    content += '\n'
    if formatted:
      content += ' / '.join(f'{k} {v}' for k, v in formatted.items())
    else:
      content += 'No metrics.'
    elements.print(f'\n{header}\n{content}\n', flush=True)


def _optional_int(value):
  if value in (None, '', 'null', 'None'):
    return None
  return int(value)


def _resolve_seed_protocol(config):
  experiment_seed = _optional_int(config.get('experiment_seed', None))
  if experiment_seed is None:
    raise ValueError('experiment_seed must be configured for walk-forward runs')

  config_seed = _optional_int(config.get('seed', None))
  raw_dreamer_seed = _optional_int(config.dreamer.get('seed', None))
  if raw_dreamer_seed is None and config_seed is None:
    raise ValueError('dreamer.seed must be configured for walk-forward runs')
  if (
      config_seed is not None and config_seed != 0 and
      raw_dreamer_seed is not None and raw_dreamer_seed != 0 and
      config_seed != raw_dreamer_seed):
    raise ValueError(
        f'config.seed ({config_seed}) and dreamer.seed ({raw_dreamer_seed}) '
        'must match')
  dreamer_seed_source = 'dreamer.seed'
  if raw_dreamer_seed in (None, 0) and config_seed not in (None, 0):
    dreamer_seed = int(config_seed)
    dreamer_seed_source = 'seed'
    elements.print(
        '[seed_protocol] --seed/config.seed is deprecated for walk-forward; '
        'using it as dreamer.seed because dreamer.seed was not set.',
        color='yellow')
  elif raw_dreamer_seed is None:
    raise ValueError('dreamer.seed must be configured for walk-forward runs')
  else:
    dreamer_seed = int(raw_dreamer_seed)

  raw_train_env_seed = _optional_int(config.env.get('train_seed', None))
  train_env_seed = raw_train_env_seed
  if train_env_seed in (None, 0):
    train_env_seed = experiment_seed * 1000 + 101

  raw_replay_seed = _optional_int(config.replay.get('seed', None))
  replay_seed = raw_replay_seed
  if replay_seed in (None, 0):
    replay_seed = experiment_seed * 1000 + 202

  eval_env_seed = _optional_int(config.env.get('eval_seed', None))
  if eval_env_seed is None:
    raise ValueError('env.eval_seed must be configured')
  matched_random_seed = _optional_int(config.audit.get('matched_random_seed', None))
  if matched_random_seed is None:
    raise ValueError('audit.matched_random_seed must be configured')

  protocol = {
      'experiment_seed': int(experiment_seed),
      'dreamer_seed': int(dreamer_seed),
      'train_env_seed': int(train_env_seed),
      'replay_seed': int(replay_seed),
      'eval_env_seed': int(eval_env_seed) if eval_env_seed is not None else None,
      'matched_random_seed': int(matched_random_seed),
      'config_seed': int(dreamer_seed),
      'config_seed_sync': 'config.seed synchronized from resolved dreamer_seed',
      'dreamer_seed_source': dreamer_seed_source,
      'train_env_seed_source': (
          'env.train_seed' if raw_train_env_seed not in (None, 0)
          else 'experiment_seed*1000+101'),
      'replay_seed_source': (
          'replay.seed' if raw_replay_seed not in (None, 0)
          else 'experiment_seed*1000+202'),
      'eval_env_seed_source': (
          'env.eval_seed' if eval_env_seed is not None
          else 'not_used'),
      'matched_random_seed_source': 'audit.matched_random_seed',
      'entry_eval_version': str(config.audit.get('entry_eval_version', '') or ''),
      'split_manifest_hash': str(config.audit.get('split_manifest_hash', '') or ''),
      'execution_timing': str(config.audit.get('execution_timing', '') or ''),
      'train_env_seed_usage': 'first_reset_only_per_env',
      'replay_seed_usage': 'replay_selector_sampling',
      'eval_env_seed_usage': 'not_used_per_day_forced_start',
      'eval_policy_mode': 'pred',
      'matched_random_seed_usage': 'fixed_across_dreamer_seeds',
      'gymnasium_constructor_seed_usage': 'forbidden',
      'eval_determinism': (
          'per_day checkpoint audits force deterministic start rows and eval '
          'policy uses pred(); eval_env_seed is recorded but not used to sample '
          'validation/test starts'),
  }
  return config.update({
      'seed': int(dreamer_seed),
      'experiment_seed': int(experiment_seed),
      'dreamer.seed': int(dreamer_seed),
      'env.train_seed': int(train_env_seed),
      'env.eval_seed': int(eval_env_seed),
      'replay.seed': int(replay_seed),
      'audit.matched_random_seed': int(matched_random_seed),
      'seed_protocol': protocol,
  })


def _env_reset_seed(config, index):
  if not config.get('seed_protocol', None):
    return None
  if config.script in ('eval_only', 'live_trading', 'monte_carlo'):
    base = config.seed_protocol.get('eval_env_seed', None)
  else:
    base = config.seed_protocol.get('train_env_seed', None)
  if base is None:
    return None
  return int(base) + int(index)

def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))
  config = _resolve_seed_protocol(config)

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  os.environ["DREAMER_RUN_DIR"] = str(logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
      action_mask_warmup=config.action_mask_warmup,
      action_mask_actor_initial=config.agent.avail_actor_enabled,
      seed_protocol=config.seed_protocol,
  )

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'offline_pretrain':
    embodied.run.offline_pretrain(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)
    
  elif config.script == 'live_trading':
    embodied.run.live_trading(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'monte_carlo':
    embodied.run.monte_carlo(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  terminal_output = (
      LiveTerminalOutput if config.script == 'live_trading'
      else elements.logger.TerminalOutput)
  outputs.append(terminal_output(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory,
      name=f'{folder}-{mode}', seed=int(config.seed_protocol.replay_seed))

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    seed = int(config.seed_protocol.replay_seed)
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(seed + 11),
        priority=selectors.Prioritized(**config.replay.prio, seed=seed + 12),
        recency=selectors.Recency(recency, seed=seed + 13),
    ), config.replay.fracs, seed=seed + 14)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  use_ctor_seed = kwargs.pop('use_seed', False)
  if suite == 'gymnasium' and use_ctor_seed:
    raise ValueError(
        'env.gymnasium.use_seed=True passes seed into env construction. '
        'Use env.train_seed/env.eval_seed for reset(seed=...) instead.')
  if use_ctor_seed:
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if suite == 'gymnasium':
    reset_seed = _env_reset_seed(config, index)
    if reset_seed is not None:
      kwargs['reset_seed'] = reset_seed
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
