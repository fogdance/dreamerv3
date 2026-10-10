"""Explicit audited C phase transfer, separate from financial-account resume."""
import hashlib
import json
import pickle
from pathlib import Path


def validate_transfer(config, receipt_path):
  from gym_trading_env.envs.entry_contract import fingerprint, read_yaml
  receipt=json.loads(Path(receipt_path).read_text())
  digest=receipt.pop('content_hash')
  if fingerprint(receipt)!=digest or receipt['method']!='C':
    raise ValueError('Invalid C transfer receipt')
  if int(config.experiment_seed)!=receipt['seed']:
    raise ValueError('C phase seed mismatch')
  target=Path(config.fg_entry.manifest).resolve()
  if target!=Path(receipt['target_manifest']).resolve():
    raise ValueError('C recent manifest mismatch')
  for filename,expected in receipt['input_hashes'].items():
    if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
      raise ValueError(f'C transfer frozen input changed: {filename}')
  source=read_yaml(receipt['source_config'])
  if int(source['experiment_seed'])!=receipt['seed'] or int(source['run']['steps'])!=receipt['source_steps']:
    raise ValueError('C source seed/budget mismatch')
  source_manifest=json.loads(Path(source['fg_entry']['manifest']).read_text())
  target_manifest=json.loads(target.read_text())
  for manifest in (source_manifest,target_manifest):
    expected=manifest.pop('content_hash')
    if fingerprint(manifest)!=expected:
      raise ValueError('C dataset manifest changed')
    validation=[r['contract'] for r in manifest['files'] if r['role']=='validation']
    if validation!=[receipt['fold']]:
      raise ValueError('C fold mismatch')
  old={r['contract'] for r in source_manifest['files'] if r['role']=='train'}
  new={r['contract'] for r in target_manifest['files'] if r['role']=='train'}
  if not new or not new<=old:
    raise ValueError('C recent data is not a subset of pretraining')
  for key in ('agent','batch_size','batch_length','train_ratio'):
    left=source['run']['train_ratio'] if key=='train_ratio' else source[key]
    right=config.run.train_ratio if key=='train_ratio' else getattr(config,key)
    if fingerprint(left)!=fingerprint(right):
      raise ValueError(f'C phase model/training structure mismatch: {key}')
  checkpoint=Path(receipt['checkpoint'])
  if not (checkpoint/'done').is_file():
    raise ValueError('C pretrain checkpoint incomplete')
  step=pickle.loads((checkpoint/'step.pkl').read_bytes())
  if int(step)!=receipt['source_steps']:
    raise ValueError('C pretrain budget incomplete')
  if Path(config.logdir).resolve()==Path(source['logdir']).resolve():
    raise ValueError('C recent requires an independent logdir')
  return receipt
