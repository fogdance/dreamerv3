"""FG training admission: explicit frozen train role and audited lifecycle only."""
import hashlib
import json
from pathlib import Path


def validate_fg_entry(config):
  if config.task not in ('gymnasium_FixedDirectionEntryEnv-v0','gymnasium_FGEntryDatasetEnv-v0'):
    return
  if config.script != 'train':
    raise ValueError('FG account checkpointing requires the ordinary train script')
  if config.run.from_checkpoint:
    raise ValueError('FG cannot restore model-only checkpoints; resume the same logdir to restore the financial account')
  from gym_trading_env.envs.entry_contract import fingerprint, read_yaml
  if getattr(config.run, 'fg_full_checkpoint', False):
    if config.task != 'gymnasium_FGEntryDatasetEnv-v0' or config.run.envs != 1:
      raise ValueError('FG full checkpoints require sequential single-env datasets')
    if config.jax.platform != 'cuda' or config.replay.fracs.uniform != 1:
      raise ValueError('FG full checkpoints require GPU and Uniform Replay')
    selector=read_yaml(config.env.gymnasium.config_path)
    if selector.get('scheduler', {}).get('mode', 'sequential') != 'sequential':
      raise ValueError('FG full checkpoints forbid rotating account scheduling')
    if int(config.run.fg_report_every_steps)<=0 or int(config.run.fg_save_every_steps)<=0:
      raise ValueError('FG fixed step periods must be positive')
    from embodied.run.fg_checkpoint import provenance, validate_bundle
    checkpoint_root=Path(config.logdir)/'ckpt'
    nodes=sorted(checkpoint_root.glob('step_*'),reverse=True)
    if nodes:
      from embodied.run.fg_checkpoint import FullCheckpoint
      # Validate before main overwrites config.yaml or initializes the GPU Agent.
      FullCheckpoint(checkpoint_root, None, provenance(config), ()).latest()
  transfer = getattr(config.run, 'fg_stage_transfer', '')
  if transfer:
    from .fg_stage_transfer import validate_transfer
    validate_transfer(config, transfer)
  path = Path(config.fg_entry.manifest).expanduser()
  if not config.fg_entry.manifest or not path.is_file():
    raise ValueError('FG training requires a frozen dataset manifest')
  manifest = json.loads(path.read_text())
  expected = manifest.pop('content_hash')
  if fingerprint(manifest) != expected or manifest['product'] != 'FG':
    raise ValueError('FG manifest integrity or product mismatch')
  roles = manifest['roles']
  if not (max(roles['train']) < min(roles['validation']) <= max(roles['validation']) < min(roles['test'])):
    raise ValueError('FG dataset roles overlap or are not chronological')
  selected = Path(config.env.gymnasium.config_path).expanduser().resolve()
  if config.task == 'gymnasium_FGEntryDatasetEnv-v0':
    selector=read_yaml(selected)
    if selector.get('strategy_mode')!='entry_dataset' or selector.get('role')!='train' or Path(selector['manifest_path']).resolve()!=path.resolve():
      raise ValueError('FG dataset selector must refer to the frozen train manifest')
    records=[r for r in manifest['files'] if r['role']=='train' and r['eligible_for_training']]
    if not records:
      raise ValueError('No complete FG training lifecycles')
    for row in records:
      validate_fg_entry(config.update({'task':'gymnasium_FixedDirectionEntryEnv-v0',
          'env.gymnasium.config_path':row['env_config'],
          'run.fg_stage_transfer':'', 'run.fg_full_checkpoint':False}))
    return
  records = [r for r in manifest['files'] if Path(r['env_config']).resolve() == selected]
  if len(records) != 1 or not records[0]['eligible_for_training'] or records[0]['role'] != 'train':
    raise ValueError('FG training requires an audited complete train lifecycle; unresolved main-roll positions cannot be reset')
  row = records[0]
  pairs = [(row['path'],row['sha256']), (row['env_config'],row['env_sha256']),
      (row['contract_config'],row['contract_sha256']),
      (manifest['direction_path'],manifest['direction_sha256'])]
  for filename, digest in pairs:
    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
      raise ValueError(f'FG frozen input changed: {filename}')
  env = read_yaml(selected)
  from gym_trading_env.envs.entry_contract import resolve_path
  if (Path(env['trading']['data_path']).expanduser().resolve() != Path(row['path']).resolve()
      or resolve_path(selected,env['entry'].get('strategy_config',env['entry'].get('direction_config'))) != Path(manifest['direction_path']).resolve()
      or resolve_path(selected,env['entry']['contract_config']) != Path(row['contract_config']).resolve()):
    raise ValueError('FG environment references unfrozen dataset inputs')
  if env['strategy_mode'] != 'entry_only' or env['entry']['contract'] != row['contract']:
    raise ValueError('FG environment does not match the frozen dataset')
  if not config.run.env_checkpoint or config.agent.avail_always_available_index != 0:
    raise ValueError('FG requires account checkpoints and an always available WAIT')
