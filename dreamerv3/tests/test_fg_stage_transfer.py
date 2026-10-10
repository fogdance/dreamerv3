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
