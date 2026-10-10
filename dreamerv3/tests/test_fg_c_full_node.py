import json
from pathlib import Path
import elements
import pytest
import yaml
from embodied.run.fg_checkpoint import FullCheckpoint, learning_spec
from dreamerv3.fg_stage_transfer import validate_transfer
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.entry_pipeline import file_hash


def test_full_c_branch_accepts_actual_node_inside_longer_a_budget(tmp_path,monkeypatch):
  def manifest(name,train):
    data=dict(files=[dict(contract=c,role='train') for c in train]+[dict(contract='FG409',role='validation')])
    data['content_hash']=fingerprint(data)
    path=tmp_path/name;path.write_text(json.dumps(data));return path
  old=manifest('all.json',['FG301','FG305']);new=manifest('recent.json',['FG305'])
  source_root=tmp_path/'all';source_root.mkdir()
  cfg=dict(experiment_seed=0,seed=0,run=dict(steps=100,train_ratio=64,fg_full_checkpoint=True,
      fg_report_every_steps=5,fg_save_every_steps=10,fg_source_steps=0),fg_entry=dict(manifest=str(old)),
      agent=dict(width=512),batch_size=16,batch_length=64,logdir=str(source_root))
  source=source_root/'config.yaml';source.write_text(yaml.safe_dump(cfg))
  target=elements.Config(cfg).update({'run.steps':30,'run.fg_source_steps':70,'logdir':str(tmp_path/'recent'),'fg_entry.manifest':str(new)})
  identity=dict(learning=learning_spec(target),code_hash='code',python='python',versions={})
  monkeypatch.setattr('embodied.run.fg_checkpoint.provenance',lambda config:identity)
  step=elements.Counter(70)
  cp=FullCheckpoint(source_root/'ckpt',step,identity,(),[70],reserve_gib=0)
  cp.step=step;cp.agent=elements.Counter();cp.driver=elements.Counter();cp.runtime=elements.Counter()
  cp.action_mask_warmup=elements.Counter()
  import embodied
  cp.replay=embodied.replay.Replay(3,64,online=True)
  cp.save();node=cp.latest()
  raw=dict(method='C',fold='FG409',seed=0,source_steps=70,source_config=str(node/'config.yaml'),
      target_manifest=str(new),checkpoint=str(node),full_checkpoint=True,
      input_hashes={str(p):file_hash(p) for p in (node/'config.yaml',old,new)})
  def receipt():
    path=tmp_path/'receipt.json';path.write_text(json.dumps(dict(raw,content_hash=fingerprint(raw))));return path
  assert validate_transfer(target,receipt())['source_steps']==70
  # A's configured maximum is 100, the protected source node is exactly 70.
  raw['source_steps']=71
  with pytest.raises(ValueError,match='source step identity mismatch'):validate_transfer(target,receipt())
  raw['source_steps']=70
  changed=target.update({'run.fg_report_every_steps':6})
  with pytest.raises(ValueError,match='learning configuration'):validate_transfer(changed,receipt())


def test_c_origin_binds_parent_receipt(tmp_path):
  from embodied.run.fg_checkpoint import stage_origin
  raw=dict(method='C',seed=0,source_steps=70,target_manifest=str(tmp_path/'data.json'),checkpoint='parent70')
  path=tmp_path/'source_transfer.json'
  def write():path.write_text(json.dumps(dict(raw,content_hash=fingerprint(raw))))
  write()
  cfg=dict(run=dict(fg_source_steps=70),logdir=str(tmp_path),experiment_seed=0,fg_entry=dict(manifest=raw['target_manifest']))
  old=stage_origin(cfg)
  raw['checkpoint']='different_parent';write()
  assert stage_origin(cfg)!=old
  cfg['run']['fg_source_steps']=700
  with pytest.raises(ValueError,match='source step identity'):stage_origin(cfg)
