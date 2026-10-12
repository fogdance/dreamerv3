import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import yaml
import pytest
from dreamerv3.fg_stage_transfer import validate_transfer
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.entry_pipeline import file_hash


def test_c_transfer_requires_completed_budget_same_seed_fold_and_model(tmp_path):
  def manifest(name,train):
    raw=dict(files=[dict(contract=c,role='train') for c in train]+[dict(contract='FG409',role='validation')])
    raw['content_hash']=fingerprint(raw)
    p=tmp_path/name;p.write_text(json.dumps(raw));return p
  old=manifest('old.json',['FG301','FG305'])
  new=manifest('new.json',['FG305'])
  source=tmp_path/'source.yaml'
  source.write_text(yaml.safe_dump(dict(experiment_seed=0,run=dict(steps=70,train_ratio=64),
      fg_entry=dict(manifest=str(old)),agent=dict(width=512,eps=1e-20),batch_size=16,batch_length=64,logdir=str(tmp_path/'old'))))
  source.write_text(source.read_text().replace('1.0e-20','1e-20'))
  checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
  (checkpoint/'done').write_text('');(checkpoint/'step.pkl').write_bytes(pickle.dumps(70))
  raw=dict(method='C',fold='FG409',seed=0,source_steps=70,source_config=str(source),
      target_manifest=str(new),checkpoint=str(checkpoint),input_hashes={str(p):file_hash(p) for p in (source,old,new)})
  def receipt():
    p=tmp_path/'receipt.json';p.write_text(json.dumps(dict(raw,content_hash=fingerprint(raw))));return p
  cfg=SimpleNamespace(experiment_seed=0,fg_entry=SimpleNamespace(manifest=str(new)),
      run=SimpleNamespace(train_ratio=64),agent=dict(width=512,eps=1e-20),batch_size=16,batch_length=64,logdir=str(tmp_path/'recent'))
  assert validate_transfer(cfg,receipt())['source_steps']==70
  cfg.agent=dict(width=256)
  with pytest.raises(ValueError,match='structure mismatch'):
    validate_transfer(cfg,receipt())
  cfg.agent=dict(width=512,eps=1e-20)
  (checkpoint/'step.pkl').write_bytes(pickle.dumps(60))
  with pytest.raises(ValueError,match='budget incomplete'):
    validate_transfer(cfg,receipt())
  (checkpoint/'step.pkl').write_bytes(pickle.dumps(70))
  raw['fold']='FG405'
  with pytest.raises(ValueError,match='fold mismatch'):
    validate_transfer(cfg,receipt())


def test_single3_completed_offline_transfer_preserves_model_and_rejects_partial_budget(tmp_path):
  import elements
  import numpy as np
  from dreamerv3.configuration import read_config_yaml
  from dreamerv3.main import _resolve_seed_protocol
  from embodied.tests.test_fg_full_checkpoint import checkpoint, State
  from embodied.run.fg_checkpoint import learning_spec
  definitions=read_config_yaml(Path(__file__).parents[1]/'configs.yaml')
  old=tmp_path/'C1_M1_OFFLINE_10K';old.mkdir()
  manifest=tmp_path/'train.json';manifest.write_text('{}')
  source=_resolve_seed_protocol(elements.Config(definitions['defaults'])
      .update(definitions['fg_entry_v1']).update(definitions['fg_single3_joint_offline'])
      .update(definitions['fg_single3_1m']).update({'logdir':str(old),
          'fg_entry.manifest':str(manifest),'fg_single3_offline.profile':'C1_M1_OFFLINE_10K','run.steps':10000}))
  identity=dict(mode='world_model_only_actor_critic_frozen',gradient_updates=10000,
      learning=learning_spec(source),candidate='C1',pool='C1_natural_oracle',code_sha256={},packages={})
  identity_hash=fingerprint(identity)
  tag=old/'single3_identity.json'
  tag.write_text(json.dumps(dict(identity=identity,identity_hash=identity_hash)))
  source_file=tmp_path/'source.yaml';source_file.write_text(yaml.safe_dump(dict(source)))
  cp,step=checkpoint(old,dict(protocol='FG_SINGLE3_OFFLINE_V1',identity_hash=identity_hash))
  audit=dict(last_update=10000,gate_open_updates=0,actor_effective_updates=0,
      critic_effective_updates=0,actor_changes_gate_closed=0)
  cp.agent=State(dict(counters=dict(updates=10000),params={'avail_actor_gate/value':np.float32(0)}))
  cp.training_audit=State(audit);cp.action_mask_warmup=State(dict(actor_enabled=False))
  step.load(10000);cp.save(final=True)
  bundle=cp.latest()
  log=old/'train_updates.jsonl';log.write_text('fixture evidence\n')
  summary=old/'training_summary.json'
  summary.write_text(json.dumps(dict(audit,status='TRAINING_COMPLETE',identity_hash=identity_hash,
      checkpoint=str(bundle),checkpoint_manifest_sha256=file_hash(bundle/'manifest.json'),log_sha256=file_hash(log))))
  target=source.update({'script':'train','task':'gymnasium_FGEntryDatasetEnv-v0',
      'logdir':str(tmp_path/'online'),'fg_single3_offline.enabled':False,'run.fg_single3_offline':False,
      'run.env_checkpoint':True,'run.fg_full_checkpoint':True,'run.fg_source_steps':10000,
      'run.steps':1000000,'replay.online':True})
  raw=dict(method='C',source_kind='single3_offline',full_checkpoint=True,seed=0,
      source_steps=10000,source_step_unit='gradient_updates',target_step_unit='environment_steps',
      source_identity_hash=identity_hash,source_config=str(source_file),target_manifest=str(manifest),
      checkpoint=str(bundle),input_hashes={str(p):file_hash(p) for p in (source_file,manifest,tag,summary)})
  receipt=tmp_path/'transfer.json'
  def write():
    receipt.write_text(json.dumps(dict(raw,content_hash=fingerprint(raw))))
  write()
  assert validate_transfer(target,receipt)['source_steps']==10000
  with pytest.raises(ValueError,match='structure mismatch'):
    validate_transfer(target.update({'agent.horizon':334}),receipt)
  raw['source_steps']=9900;write()
  with pytest.raises(ValueError,match='phase configuration mismatch'):
    validate_transfer(target,receipt)
