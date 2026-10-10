"""Synthetic FG wiring smoke: not a market backtest or timing-performance claim."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import elements
import embodied
import gym_trading_env
from ruamel import yaml
from embodied.envs.from_gymnasium import FromGymnasium
from dreamerv3.agent import Agent
from dreamerv3.main import make_stream


def main():
  source=Path(gym_trading_env.__file__).resolve().parents[2]/'tests/entry_v1/test_entry_contract.py'
  spec=importlib.util.spec_from_file_location('entry_fixture',source)
  fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
  root=Path(tempfile.mkdtemp(prefix='fg-dreamer-smoke-'))
  prices=[(1000,1000,1000,1000)]*20
  env=FromGymnasium(fixture.make_env(root,prices,main_exit='2026-01-05 09:50'))
  replay=embodied.Replay(length=4,capacity=100,chunksize=8,seed=0)
  driver=embodied.Driver([lambda:env],parallel=False)
  driver.on_step(replay.add)
  baseline=lambda carry,obs:(carry,{'action':obs['action_mask'][:,1].astype(np.int32)}, {})
  driver(baseline,steps=12)
  raw=yaml.YAML(typ='safe').load(Path('dreamerv3/configs.yaml').read_text())
  config=elements.Config(raw['defaults']).update(raw['debug']).update({
      'logdir':str(root/'model'),'batch_size':1,'batch_length':4,'report_length':4,
      'replay_context':0,'jax.compute_dtype':'float32','jax.debug':False,
      'agent.avail_always_available_index':0,'agent.avail_actor_enabled':True})
  agent=Agent({k:v for k,v in env.obs_space.items() if not k.startswith('log/')},
      {k:v for k,v in env.act_space.items() if k!='reset'},elements.Config(
      **config.agent,logdir=config.logdir,seed=0,jax=config.jax,batch_size=1,batch_length=4,
      report_length=4,replay_context=0,replica=0,replicas=1))
  carry=agent.init_train(1)
  stream=iter(agent.stream(make_stream(config,replay,'train')))
  carry,_,_=agent.train(carry,next(stream))
  _,_,metrics=agent.train(carry,next(stream))
  numbers={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
  if not numbers or not all(np.isfinite(v) for v in numbers.values()):
    raise RuntimeError('Non-finite FG training losses')
  obs=env._obs(env.env.unwrapped._get_obs(),0)
  obs={k:np.stack([v]) for k,v in obs.items() if not k.startswith('log/')}
  _,actions,_=agent.policy(agent.init_policy(1),obs,mode='eval')
  action=int(actions['action'][0])
  if not obs['action_mask'][0,action]:
    raise RuntimeError('Dreamer produced a masked real FG action')
  result=dict(synthetic=True,formal_backtest=False,train_updates=2,action=action,
      action_mask=obs['action_mask'].tolist(),finite_metrics=len(numbers),metrics=numbers)
  path=root/'result.json';path.write_text(json.dumps(result,indent=2))
  print(json.dumps(dict(result_path=str(path),synthetic=True,train_updates=2,
      action=action,finite_metrics=len(numbers))))
  driver.close()


if __name__=='__main__':
  main()
