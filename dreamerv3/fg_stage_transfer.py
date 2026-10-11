"""Explicit audited C phase transfer, separate from financial-account resume."""
import json
import pickle
from pathlib import Path


def validate_transfer(config, receipt_path):
  from gym_trading_env.envs.entry_contract import fingerprint
  from .configuration import read_config_yaml
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
  source=read_config_yaml(receipt['source_config'])
  if receipt.get('source_kind')=='single3_offline':
    return _validate_single3_transfer(config,receipt,source)
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


def _validate_single3_transfer(config, receipt, source):
  """Transfer a completed frozen-Gate pretraining bundle to online TRAIN."""
  from gym_trading_env.envs.entry_contract import fingerprint
  from embodied.run.fg_checkpoint import validate_bundle, file_hash, learning_spec
  import importlib.metadata
  budget=receipt['source_steps']
  if (not receipt.get('full_checkpoint') or not config.run.fg_full_checkpoint
      or not config.run.env_checkpoint or config.run.envs!=1
      or config.script!='train' or config.task!='gymnasium_FGEntryDatasetEnv-v0'
      or config.fg_single3_offline.enabled or config.run.fg_single3_offline
      or source['script']!='offline_pretrain' or not source['fg_single3_offline']['enabled']
      or source['fg_single3_offline']['profile']!='C1_M1_OFFLINE_10K'
      or budget!=10000 or source['run']['steps']!=budget or config.run.fg_source_steps!=budget
      or config.run.steps!=1000000 or config.run.train_ratio!=64
      or not config.replay.online or config.replay.fracs.uniform!=1
      or config.agent.avail_actor_enabled or not config.action_mask_warmup.enabled
      or receipt.get('source_step_unit')!='gradient_updates'
      or receipt.get('target_step_unit')!='environment_steps'):
    raise ValueError('Single3 offline-to-online phase configuration mismatch')
  if Path(source['fg_entry']['manifest']).resolve()!=Path(config.fg_entry.manifest).resolve():
    raise ValueError('Single3 online must use the same frozen TRAIN manifest')
  for key in ('agent','jax','seed_protocol','action_mask_warmup','batch_size',
      'batch_length','report_length','replay_context','consec_train','consec_report'):
    if fingerprint(source[key])!=fingerprint(getattr(config,key)):
      raise ValueError(f'Single3 transfer model/training structure mismatch: {key}')
  if source['run']['train_ratio']!=config.run.train_ratio:
    raise ValueError('Single3 transfer train ratio changed')
  source_root=Path(source['logdir'])
  if source_root.resolve()==Path(config.logdir).resolve():
    raise ValueError('Single3 online requires an independent logdir')
  tag=json.loads((source_root/'single3_identity.json').read_text())
  identity=tag['identity']
  expected_hash=receipt['source_identity_hash']
  if (tag['identity_hash']!=expected_hash or fingerprint(identity)!=expected_hash
      or identity['mode']!='world_model_only_actor_critic_frozen'
      or identity['gradient_updates']!=budget
      or fingerprint(identity['learning'])!=fingerprint(learning_spec(source))
      or identity['candidate']!='C1' or identity['pool']!='C1_natural_oracle'):
    raise ValueError('Single3 source identity mismatch')
  root=Path(__file__).parents[1]
  for name, expected in identity['code_sha256'].items():
    if file_hash(root/name)!=expected:
      raise ValueError(f'Single3 source code changed: {name}')
  for name, version in identity['packages'].items():
    if importlib.metadata.version(name)!=version:
      raise ValueError(f'Single3 source package changed: {name}')
  checkpoint=Path(receipt['checkpoint'])
  meta=validate_bundle(checkpoint,dict(protocol='FG_SINGLE3_OFFLINE_V1',identity_hash=expected_hash))
  summary=json.loads((source_root/'training_summary.json').read_text())
  audit=pickle.loads((checkpoint/'training_audit.pkl').read_bytes())
  warmup=pickle.loads((checkpoint/'action_mask_warmup.pkl').read_bytes())
  if (meta['step']!=budget or not meta['protected']
      or summary['status']!='TRAINING_COMPLETE' or summary['last_update']!=budget
      or summary['identity_hash']!=expected_hash
      or Path(summary['checkpoint']).resolve()!=checkpoint.resolve()
      or summary['checkpoint_manifest_sha256']!=file_hash(checkpoint/'manifest.json')
      or summary['log_sha256']!=file_hash(source_root/'train_updates.jsonl')
      or any(summary.get(k)!=v for k,v in audit.items())
      or any(audit[k] for k in ('gate_open_updates','actor_effective_updates','critic_effective_updates','actor_changes_gate_closed'))
      or warmup['actor_enabled']):
    raise ValueError('Single3 source pretraining incomplete or Actor/Critic activated')
  agent=pickle.loads((checkpoint/'agent.pkl').read_bytes())
  if agent['counters']['updates']!=budget or bool(agent['params']['avail_actor_gate/value']):
    raise ValueError('Single3 source Agent counter/Gate mismatch')
  return receipt
