"""Explicit audited C phase transfer, separate from financial-account resume."""
import json
import pickle
from pathlib import Path


def validate_transfer(config, receipt_path):
  from gym_trading_env.envs.entry_contract import fingerprint
  from ruamel import yaml
  receipt=json.loads(Path(receipt_path).read_text())
  digest=receipt.pop('content_hash')
  if getattr(config.run, 'fg_full_checkpoint', False) and not receipt.get('full_checkpoint'):
    raise ValueError('FG full target cannot migrate a legacy incomplete checkpoint')
  if fingerprint(receipt)!=digest or receipt['method']!='C':
    raise ValueError('Invalid C transfer receipt')
  if int(config.experiment_seed)!=receipt['seed']:
    raise ValueError('C phase seed mismatch')
  target=Path(config.fg_entry.manifest).resolve()
  if target!=Path(receipt['target_manifest']).resolve():
    raise ValueError('C recent manifest mismatch')
  from embodied.run.fg_checkpoint import file_hash
  for filename,expected in receipt['input_hashes'].items():
    if file_hash(filename)!=expected:
      raise ValueError(f'C transfer frozen input changed: {filename}')
  source=yaml.YAML(typ='safe').load(Path(receipt['source_config']).read_text())
  if int(source['experiment_seed'])!=receipt['seed'] or (not receipt.get('full_checkpoint') and int(source['run']['steps'])!=receipt['source_steps']):
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
  if receipt.get('full_checkpoint'):
    from embodied.run.fg_checkpoint import validate_bundle, learning_spec, digest, provenance
    if not getattr(config.run, 'fg_full_checkpoint', False):
      raise ValueError('C full source requires a full-state target')
    if int(getattr(config.run, 'fg_source_steps', 0)) != receipt['source_steps']:
      raise ValueError('C source step identity mismatch')
    meta=validate_bundle(checkpoint)
    if not meta['protected']:
      raise ValueError('C source must be a protected immutable branch node')
    if digest(learning_spec(source)) != digest(learning_spec(config)):
      raise ValueError('C full source learning configuration mismatch')
    target_identity=provenance(config)
    for key in ('learning','code_hash','python','versions'):
      if meta['identity'][key]!=target_identity[key]:
        raise ValueError(f'C full source runtime identity mismatch: {key}')
  if not (checkpoint/'done').is_file():
    raise ValueError('C pretrain checkpoint incomplete')
  step=pickle.loads((checkpoint/'step.pkl').read_bytes())
  if int(step)!=receipt['source_steps']:
    raise ValueError('C pretrain budget incomplete')
  if Path(config.logdir).resolve()==Path(source['logdir']).resolve():
    raise ValueError('C recent requires an independent logdir')
  return receipt
