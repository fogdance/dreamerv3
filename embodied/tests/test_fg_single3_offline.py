"""CPU FakeAgent pilot wiring and full-state checkpoint evidence; no neural Agent."""
from functools import partial
from pathlib import Path

import elements
import embodied
import numpy as np
import pytest
import yaml

from embodied.run.offline_pretrain import offline_pretrain
from embodied.run.fg_checkpoint import validate_bundle
from dreamerv3.fg_single3_offline import validate


class FakeAgent:
  def __init__(self):self.actor=False;self.calls=[]
  def init_train(self,n):return np.zeros(n,np.int32)
  init_report=init_train
  def stream(self,stream):return embodied.streams.Prefetch(stream)
  def train(self,carry,batch):
    self.calls.append(self.actor)
    return carry+1,{},dict(loss=np.float32(1),**{
      'avail/post_accuracy':1.,'avail/post_exact_accuracy':1.,'avail/prior_accuracy':1.,
      'avail/prior_exact_accuracy':1.,'avail/prior_fpr':0.,'avail/img_fallback_rate':0.,
      'avail/post_add_positives':32.,'avail/post_add_true_positives':32.,
      'avail/prior_add_positives':32.,'avail/prior_add_true_positives':32.})
  def report(self,carry,batch):return carry+1,{'report':1.}
  def set_avail_actor_enabled(self,value):self.actor=bool(value)
  def get_avail_actor_enabled(self):return self.actor
  def save(self):return dict(actor=self.actor,calls=self.calls)
  def load(self,state):self.actor=state['actor'];self.calls=state['calls']
  load_exact=load


def defaults():
  return elements.Config(yaml.safe_load((Path(__file__).parents[2]/'dreamerv3/configs.yaml').read_text())['defaults'])


def test_single3_unapproved_and_legacy_entry_fail_before_agent():
  config=defaults().update({'fg_single3_offline.enabled':True})
  with pytest.raises(ValueError,match='NOT_RUN'):
    validate(config)
  with pytest.raises(ValueError,match='joint offline'):
    validate(config.update({'fg_single3_offline.authorized':True}))


def test_joint_offline_reuses_add_recall_gate_and_full_checkpoint(tmp_path):
  cfg=defaults()
  gate=cfg.action_mask_warmup.update(enabled=True,add_recall_enabled=True,
      min_train_samples=1,consecutive_reports=1)
  directory=tmp_path/'source'
  replay=embodied.Replay(length=3,capacity=64,chunksize=16,online=False,directory=str(directory),seed=202)
  for i in range(12):
    replay.add(dict(value=np.float32(i),reward=np.float32(0),is_first=np.bool_(i==0),
        is_last=np.bool_(False),is_terminal=np.bool_(False)))
  replay.save()
  replay.workers.shutdown(wait=True)
  args=elements.Config(dict(cfg.run) | dict(logdir=str(tmp_path/'run'),batch_size=1,batch_length=2,
      consec_report=1,steps=2,fg_single3_offline=True,fg_single3_identity='fake_test_identity',
      action_mask_warmup=gate,action_mask_actor_initial=False,fg_report_every_steps=1,
      fg_save_every_steps=1,fg_key_steps=[0,2],fg_reserve_gib=0,
      usage=dict(psutil=False,nvsmi=False,gputil=False,malloc=False,gc=False)))
  agents=[]
  def make_agent():
    result=FakeAgent();agents.append(result);return result
  def make_replay():return embodied.Replay(length=3,capacity=64,chunksize=16,online=False,seed=202,
      directory=str(tmp_path/'run/replay'))
  def make_stream(buffer,mode):return embodied.streams.Stateless(partial(buffer.sample,1,mode=mode))
  def make_logger():return elements.Logger(elements.Counter(),[lambda _: None])
  offline_pretrain(make_agent,make_replay,make_stream,make_logger,args,replay_dir=str(directory))
  assert agents[-1].calls==[False,True]
  meta=validate_bundle(tmp_path/'run/ckpt/step_000000000002')
  assert meta['identity']['protocol']=='FG_SINGLE3_OFFLINE_V1' and meta['protected']
  offline_pretrain(make_agent,make_replay,make_stream,make_logger,args,replay_dir=str(directory))
  assert agents[-1].calls==[False,True] and agents[-1].actor  # no duplicate updates on resume
  with pytest.raises(ValueError,match='identity mismatch'):
    offline_pretrain(make_agent,make_replay,make_stream,make_logger,
        args.update(fg_single3_identity='other_reward'),replay_dir=str(directory))
  replay.workers.shutdown(wait=True)
