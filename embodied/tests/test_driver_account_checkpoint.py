import embodied
import elements
import numpy as np
import pytest


class Account:
  def __init__(self):
    self.balance=300000
    self.position=0
  @property
  def act_space(self):
    return {'action':elements.Space(np.int32,(),0,2),'reset':elements.Space(bool)}
  def step(self,action):
    if not action['reset']:
      self.position+=int(action['action'])
      self.balance-=17
    return dict(balance=np.int32(self.balance),position=np.int32(self.position),
                reward=np.float32(0),is_first=bool(action['reset']),is_last=False,is_terminal=False)
  def save(self):
    return dict(balance=self.balance,position=self.position)
  def load(self,state):
    self.balance,self.position=state['balance'],state['position']
  def close(self):
    pass


def policy(carry,obs):
  return carry+1,{'action':np.ones(1,np.int32)},{}


@pytest.mark.parametrize('parallel',[False,True])
def test_driver_resumes_pending_actions_and_account(parallel):
  driver=embodied.Driver([Account],parallel=parallel)
  clone=embodied.Driver([Account],parallel=parallel)
  try:
    driver.reset(lambda n:np.zeros(n,np.int32))
    driver(policy,steps=3)
    state=driver.save()
    clone.load(state)
    left,right=[],[]
    driver.on_step(lambda t,i:left.append(t))
    clone.on_step(lambda t,i:right.append(t))
    driver(policy,steps=3)
    clone(policy,steps=3)
    for a,b in zip(left,right):
      for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    assert driver.save()['envs']==clone.save()['envs']
    assert driver.save()['envs'][0]['position']==5
  finally:
    driver.close();clone.close()
