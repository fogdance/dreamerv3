from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np

def render_only(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  policy_fps = elements.FPS()


  fns = [bind(make_env, i) for i in range(1)]
  driver = embodied.Driver(fns, False)
  env = driver.envs[0]
  driver.on_step(lambda tran, _: env.render())
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())

  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])

  print("Start rendering...")
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  driver(policy, episodes=1)

  logger.close()
