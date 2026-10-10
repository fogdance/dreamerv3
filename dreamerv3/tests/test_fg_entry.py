from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest
from dreamerv3.agent import hard_nonempty_mask
from dreamerv3.fg_entry_guard import validate_fg_entry


def test_uncertain_two_action_mask_falls_back_to_wait():
  mask, used=hard_nonempty_mask(jnp.array([[.1,.4],[.1,.9]]),.5,0)
  np.testing.assert_array_equal(mask,[[True,False],[True,True]])
  np.testing.assert_array_equal(used,[True,False])
  legacy,_=hard_nonempty_mask(jnp.array([[.1,.4]]),.5)
  np.testing.assert_array_equal(legacy,[[False,True]])


def test_fg_cannot_train_without_manifest():
  config=SimpleNamespace(task='gymnasium_FixedDirectionEntryEnv-v0',script='train',
      run=SimpleNamespace(from_checkpoint=''),fg_entry=SimpleNamespace(manifest=''))
  with pytest.raises(ValueError,match='frozen dataset'):
    validate_fg_entry(config)
  validate_fg_entry(SimpleNamespace(task='gymnasium_CustomTradingEnv-v0'))
  config.run.from_checkpoint='model-only-checkpoint'
  with pytest.raises(ValueError,match='financial account'):
    validate_fg_entry(config)


def test_real_gym_adapter_restores_losing_account_and_wrapper(tmp_path):
  import importlib.util
  import pathlib
  import gymnasium as gym
  from embodied.envs.from_gymnasium import FromGymnasium
  import gym_trading_env
  source=pathlib.Path(gym_trading_env.__file__).resolve().parents[2]/'tests/entry_v1/test_entry_contract.py'
  if not source.exists():
    pytest.skip('Requires companion environment test checkout')
  spec=importlib.util.spec_from_file_location('entry_fixture',source)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
  bars=[(1000,1000,1000,1000),(1000,1000,1000,1000),(990,990,990,990),(980,980,980,980)]
  left=FromGymnasium(gym.wrappers.OrderEnforcing(module.make_env(first,bars)))
  right=FromGymnasium(gym.wrappers.OrderEnforcing(module.make_env(second,bars)))
  left.step({'action':0,'reset':True})
  left.step({'action':1,'reset':False})
  left.step({'action':0,'reset':False})
  right.load(left.save())
  a=left.step({'action':0,'reset':False})
  b=right.step({'action':0,'reset':False})
  for k in a:
    np.testing.assert_array_equal(a[k],b[k])
  assert right.env.unwrapped.net_lots==3
  assert right.env.unwrapped._get_info()['account_net_pnl']==-1200
