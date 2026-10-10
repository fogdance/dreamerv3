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
  path = Path(config.fg_entry.manifest).expanduser()
  if not config.fg_entry.manifest or not path.is_file():
    raise ValueError('FG training requires a frozen dataset manifest')
  manifest = json.loads(path.read_text())
  expected = manifest.pop('content_hash')
  if fingerprint(manifest) != expected or manifest['product'] != 'FG':
    raise ValueError('FG manifest integrity or product mismatch')
  validate_manifest_roles(manifest)
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
          'env.gymnasium.config_path':row['env_config']}))
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


def validate_manifest_roles(manifest):
  """Accept frozen whole-contract groups and retain the legacy chronology gate."""
  roles=manifest['roles']
  if set(roles)!={'train','validation','test'} or not all(roles.values()):
    raise ValueError('FG dataset requires three nonempty roles')
  mode=manifest.get('split',{}).get('mode','main_contract_boundaries')
  if mode=='main_contract_boundaries':
    if not (max(roles['train']) < min(roles['validation']) <= max(roles['validation']) < min(roles['test'])):
      raise ValueError('FG dataset roles overlap or are not chronological')
    return
  if mode!='contract_groups' or manifest.get('schema')!=2:
    raise ValueError('Unsupported FG dataset split')
  groups=manifest['split']['contracts']
  if set(groups)!={'train','validation','test','excluded'}:
    raise ValueError('Invalid FG contract group roles')
  from gym_trading_env.envs.entry_contract import is_fg
  assigned=[c for contracts in groups.values() for c in contracts]
  if not all(is_fg(c) for c in assigned):
    raise ValueError('FG contract groups contain another product')
  if len(assigned)!=len(set(assigned)) or not all(groups[r] for r in roles):
    raise ValueError('FG contract groups overlap or are empty')
  order=manifest['split'].get('training_order',{})
  if order.get('mode')!='shuffle_each_epoch' or not isinstance(order.get('seed'),int):
    raise ValueError('FG contract groups require a frozen shuffle seed')
  actual={r:[] for r in roles}
  for row in manifest['files']:
    role=row['role']
    if role not in actual or row['contract'] not in groups[role]:
      raise ValueError('FG contract record crosses frozen roles')
    if not row['calendar_complete'] or not row['direction_rows']:
      raise ValueError('FG grouped contract lacks calendar or manual direction')
    if role=='train' and (not row['complete_lifecycle'] or not row['eligible_for_training']):
      raise ValueError('FG train contract lifecycle is incomplete')
    actual[role].append(row['contract'])
  if any(sorted(actual[r])!=sorted(groups[r]) for r in roles):
    raise ValueError('FG contract records do not cover the frozen groups exactly once')
  if sorted(r['contract'] for r in manifest['excluded_contracts'])!=sorted(groups['excluded']):
    raise ValueError('FG excluded contract audit does not match frozen groups')
  assigned_days=[day for values in roles.values() for day in values]
  if len(assigned_days)!=len(set(assigned_days)):
    raise ValueError('FG grouped trading days overlap')
