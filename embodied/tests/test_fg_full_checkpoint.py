"""Non-neural regressions; these tests never initialize a Dreamer model."""
import contextlib
import copy
import pickle
from functools import partial
from pathlib import Path

import elements
import numpy as np
import pytest

import embodied
from embodied.run.fg_checkpoint import FullCheckpoint, RuntimeState, StepSchedule, save_replay, load_replay, validate_bundle


class State:
  def __init__(self, value=0): self.value=value
  def save(self): return self.value
  def load(self, value): self.value=value
  load_exact=load


def replay():
  result=embodied.replay.Replay(length=3,capacity=64,chunksize=16,online=True,seed=7)
  for i in range(12):
    result.add(dict(value=np.float32(i),is_first=np.bool_(i==0),is_last=np.bool_(False),reward=np.float32(i)))
  result.sample(2)
  return result


def test_replay_restores_partial_chunks_online_queue_rng_and_updates(tmp_path):
  original=replay()
  batch=original.sample(1)
  original.update(dict(stepid=batch['stepid'],value=np.full((1,3),-3,np.float32)))
  directory=tmp_path/'replay'
  state=save_replay(original,directory)
  restored=replay();load_replay(restored,state,directory)
  assert restored.current==original.current and restored.itemid==original.itemid
  for _ in range(8):
    a,b=original.sample(2),restored.sample(2)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
  step=dict(value=np.float32(90),is_first=np.bool_(False),is_last=np.bool_(False),reward=np.float32(0))
  original.add(step);restored.add(step)
  assert restored.current==original.current
  for key in original.chunks:
    assert restored.chunks[key].size==16
    assert restored.chunks[key].length==original.chunks[key].length


def test_prefetch_snapshot_restores_pending_batches_and_consecutive_context(tmp_path):
  original=replay();counter=[0]
  def transform(batch):
    result={**batch,'seed':np.array(counter[0])};counter[0]+=1;return result
  def stream(buffer):
    return embodied.streams.Prefetch(embodied.streams.Consec(
        embodied.streams.Stateless(partial(buffer.sample,1)),length=1,consec=2,prefix=1),transform)
  source=stream(original);iter(source);next(source)
  with source.pause():
    state=source.save_full();saved_counter=counter[0]
    rs=save_replay(original,tmp_path/'replay')
  expected=[next(source) for _ in range(5)]
  restored=replay();load_replay(restored,rs,tmp_path/'replay');counter[0]=saved_counter
  target=stream(restored);target.load_full(state);iter(target)
  for a in expected:
    b=next(target)
    for key in a: np.testing.assert_array_equal(a[key],b[key])


def test_runtime_restores_ratio_carry_aggregates_and_gate_samples():
  schedule=StepSchedule(5,10)
  schedule.advance(5)
  ratio=elements.when.Ratio(0.5);ratio(3);ratio(8)
  agg=elements.Agg();agg.add('metric',0.9);agg.add('metric',0.7)
  carry=[np.array([8])];samples=[250]
  def state():
    return RuntimeState(schedule,ratio,(),lambda:carry[0],lambda x:carry.__setitem__(0,x),dict(gate=agg),samples)
  saved=state().save();expected=ratio(11)
  ratio._prev=None;schedule.next_report=999;carry[0]=np.array([0]);samples[0]=0;agg.reset()
  state().load(saved)
  assert ratio(11)==expected and schedule.next_report==10 and samples[0]==250
  assert agg.result()['metric']==pytest.approx(.8)
  np.testing.assert_array_equal(carry[0],[8])


def test_exact_schedule_resume_and_overshoot_rejection():
  schedule=StepSchedule(5,10,[7,20])
  assert schedule.distance(0,23)==5
  assert schedule.advance(5)==(True,False)
  assert schedule.distance(5,23)==2
  assert schedule.advance(7)==(False,True)
  saved=schedule.save();restored=StepSchedule(5,10,[7,20]);restored.load(saved)
  assert restored.distance(7,23)==3
  assert restored.advance(10)==(True,True)
  with pytest.raises(ValueError,match='overshot'):restored.advance(16)


def checkpoint(tmp_path, identity=None):
  step=elements.Counter();cp=FullCheckpoint(tmp_path/'ckpt',step,identity or dict(seed=0),(),[7],reserve_gib=0)
  cp.step=step;cp.agent=State();cp.driver=State();cp.replay=replay();cp.runtime=State();cp.action_mask_warmup=State()
  return cp,step


def test_atomic_full_bundle_retention_identity_and_corrupt_recovery(tmp_path):
  cp,step=checkpoint(tmp_path)
  for n in (0,7,10,20,30):
    step.load(n);cp.agent.value=n;cp.save()
  assert sorted(p.name for p in (tmp_path/'ckpt').glob('step_*'))==[
      'step_000000000007','step_000000000020','step_000000000030']
  protected=tmp_path/'ckpt/step_000000000007'
  assert validate_bundle(protected)['protected']
  bad,_=checkpoint(tmp_path,dict(seed=1))
  with pytest.raises(ValueError,match='identity mismatch'):bad.load()
  (cp.latest()/'agent.pkl').write_bytes(b'corrupted')
  cp.load();assert cp.agent.value==20 and int(step)==20
  # A failed pending write is never a resume candidate or deleted as old data.
  pending=tmp_path/'ckpt/.pending_test';pending.mkdir();(pending/'step.pkl').write_bytes(pickle.dumps(90))
  assert cp.latest().name=='step_000000000020'
  assert pending.exists()


def test_failed_save_preserves_valid_points_and_final_is_protected(tmp_path,monkeypatch):
  cp,step=checkpoint(tmp_path);cp.save();step.load(10);cp.save()
  previous=cp.latest()
  def fail(): raise OSError('write failed')
  monkeypatch.setattr(cp.agent,'save',fail);step.load(20)
  with pytest.raises(OSError,match='write failed'):cp.save()
  assert cp.latest()==previous
  monkeypatch.setattr(cp.agent,'save',lambda:20);cp.save(final=True)
  assert validate_bundle(cp.latest())['protected']


class CountingEnv:
  def __init__(self):
    self.act_space={'action':elements.Space(np.int32,(),0,2),'reset':elements.Space(bool)}
    self.obs_space={'value':elements.Space(np.float32),'reward':elements.Space(np.float32),
        'is_first':elements.Space(bool),'is_last':elements.Space(bool),'is_terminal':elements.Space(bool)}
    self.t=0
  def step(self,action):
    first=bool(action['reset'])
    self.t=0 if first else self.t+1
    return dict(value=np.float32(self.t),reward=np.float32(action['action']),
        is_first=np.bool_(first),is_last=np.bool_(False),is_terminal=np.bool_(False))
  def save(self):return self.t
  def load(self,t):self.t=t
  def close(self):pass


class CountingAgent:
  def __init__(self):self.updates=0;self.samples=[];self.reports=[]
  def init_policy(self,n):return np.zeros(n,np.int32)
  def init_train(self,n):return np.zeros(n,np.int64)
  def init_report(self,n):return np.zeros(n,np.int64)
  def stream(self,s):return s
  def policy(self,c,obs,mode):
    action=(obs['value'].astype(np.int32)+self.updates)%2
    return c+1,dict(action=action),{}
  def train(self,c,b):
    self.updates+=1;self.samples.append(b['value'].copy());return c+1,{},dict(loss=np.float32(self.updates))
  def report(self,c,b):
    self.reports.append(b['value'].copy());return c+1,dict(metric=np.float32(1))
  def save(self):return dict(updates=self.updates,samples=self.samples,reports=self.reports)
  def load(self,s):self.__dict__.update(s)
  load_exact=load


def test_train_loop_full_resume_matches_uninterrupted_without_neural_training(tmp_path):
  from embodied.run.train import train
  warmup=dict(enabled=False,min_train_samples=25,consecutive_reports=5,regression_reports=5,
      stop_on_regression=True,post_accuracy=.995,post_exact=.990,prior_accuracy=.990,
      prior_exact=.980,prior_fpr=.005,img_fallback=.0001)
  def run(directory,steps):
    agent=CountingAgent()
    args=elements.Config(steps=steps,train_ratio=4,log_every=1e9,report_every=1e9,save_every=1e9,
        report_batches=1,from_checkpoint='',fg_stage_transfer='',env_checkpoint=True,debug=True,
        usage=dict(psutil=False,nvsmi=False,gputil=False,malloc=False,gc=False),logdir=str(directory),
        envs=1,batch_size=1,batch_length=2,report_length=1,consec_report=1,
        action_mask_warmup=warmup,action_mask_actor_initial=False,seed_protocol={},
        fg_full_checkpoint=True,fg_report_every_steps=5,fg_save_every_steps=10,fg_key_steps=[7,13,23],
        fg_reserve_gib=0,fg_checkpoint_identity=dict(test='counting'))
    def make_replay():return embodied.replay.Replay(2,64,chunksize=32,online=True,seed=7)
    def make_stream(r,mode):return embodied.streams.Consec(
        embodied.streams.Stateless(partial(r.sample,1,mode)),length=2 if mode=='train' else 1,
        consec=1,strict=mode=='train')
    train(lambda:agent,make_replay,lambda i:CountingEnv(),make_stream,
        lambda:elements.Logger(elements.Counter(),[elements.logger.TerminalOutput()]),args)
    return agent
  expected=run(tmp_path/'continuous',23)
  run(tmp_path/'resumed',13);actual=run(tmp_path/'resumed',23)
  assert actual.updates==expected.updates
  for a,b in zip(actual.samples,expected.samples):np.testing.assert_array_equal(a,b)
  for a,b in zip(actual.reports,expected.reports):np.testing.assert_array_equal(a,b)
  assert len(actual.samples)==len(expected.samples) and len(actual.reports)==4
  nodes=tmp_path/'resumed/ckpt'
  assert (nodes/'step_000000000007').exists() and (nodes/'step_000000000013').exists()
  a=pickle.loads((nodes/'step_000000000023/driver.pkl').read_bytes())
  b=pickle.loads((tmp_path/'continuous/ckpt/step_000000000023/driver.pkl').read_bytes())
  assert a['envs']==b['envs']
  for key in a['acts']:np.testing.assert_array_equal(a['acts'][key],b['acts'][key])
