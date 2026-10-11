from pathlib import Path

import elements
import numpy as np
import pytest

from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_single3_offline import update_budget, validate
from dreamerv3.main import _resolve_seed_protocol
from embodied.run.fg_checkpoint import StepSchedule
from embodied.run.fg_single3_audit import TrainingAudit


def config():
  definitions = read_config_yaml(Path(__file__).parents[2] / 'dreamerv3/configs.yaml')
  return _resolve_seed_protocol(elements.Config(definitions['defaults']).update(definitions['fg_entry_v1'])
      .update(definitions['fg_single3_joint_offline']).update(definitions['fg_single3_1m'])
      .update({'logdir': '/tmp/C1_M1_1M'}))


def test_62500_profile_reaches_data_gate_and_rejects_changed_experiment(monkeypatch):
  from gym_trading_env.research import fg_single3_dataset
  def reached(_):
    raise RuntimeError('real data gate')
  monkeypatch.setattr(fg_single3_dataset, 'validate_dataset', reached)
  with pytest.raises(RuntimeError, match='real data gate'):
    validate(config(), allow_unapproved=True, persist=False)
  for flags in ({'run.steps': 2000}, {'run.steps': 62500.5},
      {'fg_single3_offline.pool': 'C1_natural'}, {'agent.replay_context_mode': 'stored'}):
    with pytest.raises(ValueError):
      validate(config().update(flags), allow_unapproved=True, persist=False)
  for value in (True, float('nan'), float('inf'), '62500.25', -1):
    with pytest.raises(ValueError):
      update_budget(value)
  assert update_budget('6.25e4') == 62500


def metrics(update, gate=False, changed=False):
  result = {'train/update_index': update, 'opt/finite': True, 'opt/loss': 1.,
      'opt/grad_norm': 1., 'avail/ready': float(gate)}
  for head in ('actor','critic','reward','wm'):
    result.update({'opt/' + head + '_grad_norm': 1. if gate or head in ('wm','reward') else 0.,
        'opt/' + head + '_delta_norm': float(changed), 'opt/' + head + '_changed': changed})
  return result


def test_update_evidence_counts_restore_and_nonfinite_rejection(tmp_path):
  audit = TrainingAudit(tmp_path,1024)
  audit.observe(1, metrics(1))  # zero-LR warmup does not count
  audit.observe(2, metrics(2, changed=True))  # Gate closed
  assert audit.state['actor_effective_updates'] == 0
  state = audit.save()
  resumed = TrainingAudit(tmp_path,1024)
  resumed.load(state)
  resumed.observe(3, metrics(3, gate=True, changed=True))
  assert resumed.state['actor_effective_updates'] == 1
  assert resumed.state['training_positions'] == 3072
  broken = metrics(4, gate=True, changed=True)
  broken['opt/finite'] = False
  with pytest.raises(FloatingPointError):
    resumed.observe(4, broken)
  assert resumed.state['last_update'] == 3
  with pytest.raises(ValueError, match='not completed'):
    resumed.summary('id', tmp_path,62500)
  assert not (tmp_path / 'training_summary.json').exists()


def test_required_checkpoints_are_continuous_and_protected_schedule():
  milestones = [0,2000,10000,25000,50000,62500]
  schedule = StepSchedule(100,2000,milestones)
  saved = []
  for step in range(1,62501):
    _, save = schedule.advance(step)
    if save:
      saved.append(step)
  assert set(milestones[1:]) <= set(saved)
  assert schedule.next_report == 62600


def test_audited_loop_checkpoint_restores_counter_and_gate_without_duplicate_updates(tmp_path):
  from functools import partial
  import embodied
  from embodied.run.offline_pretrain import offline_pretrain
  from embodied.tests.test_fg_single3_offline import FakeAgent,defaults
  class AuditedFake(FakeAgent):
    def __init__(self):
      super().__init__()
      self.n_updates=elements.Counter()
    def train(self,carry,batch):
      carry,outs,mets=super().train(carry,batch)
      self.n_updates.increment()
      mets.update(metrics(int(self.n_updates),self.actor,True))
      return carry,outs,mets
    def save(self):
      return dict(super().save(),updates=int(self.n_updates))
    def load(self,state):
      super().load(state)
      self.n_updates.value=state['updates']
    load_exact=load
    def drain_train(self):
      return {},{}
  base=defaults()
  source=tmp_path/'source'
  replay=embodied.Replay(length=3,capacity=64,chunksize=16,online=False,directory=str(source),seed=202)
  for i in range(12):
    replay.add(dict(value=np.float32(i),reward=np.float32(0),is_first=np.bool_(i==0),
        is_last=np.bool_(False),is_terminal=np.bool_(False)))
  replay.save();replay.workers.shutdown(wait=True)
  gate=base.action_mask_warmup.update(enabled=True,add_recall_enabled=True,min_train_samples=1,consecutive_reports=1)
  args=elements.Config(dict(base.run)|dict(logdir=str(tmp_path/'run'),batch_size=1,batch_length=2,
      consec_report=1,steps='2',fg_single3_offline=True,fg_single3_identity='audited_test',
      fg_single3_audit=True,fg_keep_all_checkpoints=True,action_mask_warmup=gate,
      action_mask_actor_initial=False,fg_report_every_steps=1,fg_save_every_steps=1,fg_key_steps=[0,2],
      fg_reserve_gib=0,usage=dict(psutil=False,nvsmi=False,gputil=False,malloc=False,gc=False)))
  agents=[]
  def factory():
    result=AuditedFake();agents.append(result);return result
  def buffer():
    return embodied.Replay(length=3,capacity=64,chunksize=16,online=False,seed=202,directory=str(tmp_path/'run/replay'))
  def stream(replay,mode):
    return embodied.streams.Stateless(partial(replay.sample,1,mode=mode))
  logger=lambda:elements.Logger(elements.Counter(),[
      elements.logger.TerminalOutput(),
      elements.logger.JSONLOutput(tmp_path/'run','metrics.jsonl')])
  offline_pretrain(factory,buffer,stream,logger,args,replay_dir=str(source))
  offline_pretrain(factory,buffer,stream,logger,args,replay_dir=str(source))
  import json
  result=json.loads((tmp_path/'run/training_summary.json').read_text())
  assert (tmp_path/'run/metrics.jsonl').is_file()
  assert int(agents[-1].n_updates)==2
  assert agents[-1].calls==[False,True]
  assert result['actor_effective_updates']==1 and result['training_positions']==4
