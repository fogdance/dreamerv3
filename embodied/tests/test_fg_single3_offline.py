"""CPU FakeAgent pilot wiring and full-state checkpoint evidence; no neural Agent."""
from functools import partial
from pathlib import Path

import elements
import embodied
import numpy as np
import pytest

from embodied.run.offline_pretrain import offline_pretrain
from embodied.run.fg_checkpoint import validate_bundle
from dreamerv3.fg_single3_offline import validate
from dreamerv3.configuration import read_config_yaml


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
  return elements.Config(read_config_yaml(Path(__file__).parents[2]/'dreamerv3/configs.yaml')['defaults'])


def test_single3_unapproved_and_legacy_entry_fail_before_agent():
  config=defaults().update({'fg_single3_offline.enabled':True})
  with pytest.raises(ValueError,match='NOT_RUN'):
    validate(config)
  with pytest.raises(ValueError,match='joint offline'):
    validate(config.update({'fg_single3_offline.authorized':True}))


def test_real_single3_preset_reaches_data_admission(monkeypatch):
  from dreamerv3.main import _resolve_seed_protocol
  from gym_trading_env.research import fg_single3_dataset
  definitions=read_config_yaml(Path(__file__).parents[2]/'dreamerv3/configs.yaml')
  config=_resolve_seed_protocol(elements.Config(definitions['defaults'])
      .update(definitions['fg_entry_v1']).update(definitions['fg_single3_joint_offline']))
  def data_gate(_):raise RuntimeError('reached real data admission')
  monkeypatch.setattr(fg_single3_dataset,'validate_dataset',data_gate)
  with pytest.raises(RuntimeError,match='reached real data admission'):
    validate(config,allow_unapproved=True,persist=False)
  with pytest.raises(ValueError,match='baseline'):
    validate(config.update({'run.steps':2001}),allow_unapproved=True,persist=False)


@pytest.mark.parametrize('profile', ('pilot_2k', 'C1_M1_1M'))
def test_real_single3_replay_capacity_is_numeric_and_exact(tmp_path, monkeypatch, profile):
  full = profile == 'C1_M1_1M'
  candidate = 'C1' if full else 'C0'
  pool = candidate + ('_natural_oracle' if full else '_natural')
  logdir = tmp_path / ('C1_M1_1M' if full else 'run')
  from dreamerv3.main import _resolve_seed_protocol, make_replay
  from gym_trading_env.research import fg_single3_dataset, fg_single3_coverage, fg_replay
  definitions=read_config_yaml(Path(__file__).parents[2]/'dreamerv3/configs.yaml')
  # Preserve compatibility with legacy configurations containing a string size.
  definitions['defaults']['replay']['size']='1e6'
  dataset=tmp_path/'dataset';dataset.mkdir()
  coverage=tmp_path/'coverage';coverage.mkdir()
  horizon=tmp_path/'horizon';horizon.mkdir()
  view=tmp_path/'view.json';view.write_text('{}')
  env=tmp_path/'env.yaml';env.write_text('entry:\n  protocol: fg_single3_v1\n')
  for root in (dataset,coverage):
    fg_replay.write_json(root/'manifest.json',fg_replay.signed({'fixture':True}))
  fg_replay.write_json(horizon/'manifest.json',fg_replay.signed(dict(stage='H',status='PASS',pairs=576,
      dataset_manifest_sha256=fg_replay.digest(dataset/'manifest.json'),
      coverage_manifest_sha256=fg_replay.digest(coverage/'manifest.json'),frozen_inputs={},files={})))
  fg_replay.write_json(dataset/'inputs.json',fg_replay.signed(dict(provenance=dict(code_files={}),
      training_views={candidate:dict(path=str(view),rows=[dict(env_config=str(env))])})))
  capacity=1949908
  data=dict(full_training_fold=True,pool_members={pool:[0]},views={pool:'pool'},
      traces=[dict(sequence_starts=capacity,reward_params_hash='fixture',observation_identity='fixture')])
  monkeypatch.setattr(fg_single3_dataset,'validate_dataset',lambda _:data)
  monkeypatch.setattr(fg_single3_coverage,'require_coverage',lambda *_:None)
  config=_resolve_seed_protocol(elements.Config(definitions['defaults'])
      .update(definitions['fg_entry_v1']).update(definitions['fg_single3_joint_offline'])
      .update(definitions['fg_single3_1m'] if full else {}).update({
          'logdir':str(logdir),'fg_entry.manifest':str(view),
          'env.gymnasium.config_path':str(env),'replay.size':capacity,
          'run.replay_dir':str(dataset/'pool'),'fg_single3_offline.dataset':str(dataset),
          'fg_single3_offline.coverage':str(coverage),'fg_single3_offline.horizon':str(horizon),
          'fg_single3_offline.candidate':candidate,'fg_single3_offline.pool':pool}))
  # Explicit legacy string size must still be validated without truncation.
  assert config.replay.size == str(capacity)
  # The launcher persists identity before the CLI checks it in another process.
  # Config sequences are tuples in memory and lists after the JSON round-trip.
  import json
  first=validate(config,allow_unapproved=True,persist=True)
  second=validate(config,allow_unapproved=True,persist=False)
  assert first==second
  assert first['identity']['mode']=='world_model_only_actor_critic_frozen'
  gate_file=Path(__file__).parents[1]/'run/action_mask_warmup.py'
  assert first['identity']['code_sha256']['embodied/run/action_mask_warmup.py']==fg_replay.digest(gate_file)
  if full:
    opening=first['identity']['mask_gate_opening']
    assert opening['thresholds']['prior_accuracy']==0.9915
    assert opening['thresholds']['prior_fpr']==0.004
    assert opening['consecutive_reports']==20 and opening['report_every_updates']==100
    assert opening['automatic_activation'] is False
    assert config.action_mask_warmup.prior_accuracy==0.99
    assert config.action_mask_warmup.prior_fpr==0.005
  else:
    assert 'mask_gate_opening' not in first['identity']
  tag=logdir/'single3_identity.json'
  stored=json.loads(tag.read_text())
  assert stored==second
  foreign=json.loads(tag.read_text())
  foreign['identity']['reward_params_hash']='other_reward'
  tag.write_text(json.dumps(foreign))
  with pytest.raises(ValueError,match='Cannot resume'):
    validate(config,allow_unapproved=True,persist=False)
  tag.write_text(json.dumps(stored))
  with pytest.raises(ValueError,match='capacity/directory'):
    validate(config.update({'replay.size':capacity+.25}),allow_unapproved=True,persist=False)
  for mode in ('train','report'):
    replay=make_replay(config,'replay_'+mode,mode=mode)
    try:
      assert replay.capacity == (capacity if mode=='train' else int(capacity/10))
      assert isinstance(replay.capacity,int) and replay.length==65
    finally:
      replay.workers.shutdown(wait=True)


def test_offline_monitors_add_recall_with_gate_closed_across_resume(tmp_path,monkeypatch):
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
      consec_report=1,steps='2',fg_single3_offline=True,fg_single3_identity='fake_test_identity',
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
  assert agents[-1].calls==[False,False] and not agents[-1].actor
  meta=validate_bundle(tmp_path/'run/ckpt/step_000000000002')
  assert meta['identity']['protocol']=='FG_SINGLE3_OFFLINE_V1' and meta['protected']
  offline_pretrain(make_agent,make_replay,make_stream,make_logger,args,replay_dir=str(directory))
  assert agents[-1].calls==[False,False] and not agents[-1].actor  # no duplicate updates on resume
  with pytest.raises(ValueError,match='identity mismatch'):
    offline_pretrain(make_agent,make_replay,make_stream,make_logger,
        args.update(fg_single3_identity='other_reward'),replay_dir=str(directory))
  load_exact=FakeAgent.load_exact
  def restore_open_gate(agent,state):
    load_exact(agent,state)
    agent.actor=True
  monkeypatch.setattr(FakeAgent,'load_exact',restore_open_gate)
  with pytest.raises(ValueError,match='closed Actor/Critic Gate'):
    offline_pretrain(make_agent,make_replay,make_stream,make_logger,
        args.update(steps='3'),replay_dir=str(directory))
  assert agents[-1].calls==[False,False]  # reject before another update
  replay.workers.shutdown(wait=True)
