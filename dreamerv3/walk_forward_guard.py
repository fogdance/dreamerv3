import csv
import hashlib
import json
from pathlib import Path

import elements
import ruamel.yaml as yaml


class WalkForwardGuardError(RuntimeError):
  pass


def _get(config, key, default=None):
  cur = config
  for part in key.split('.'):
    if cur is None:
      return default
    if hasattr(cur, 'get'):
      cur = cur.get(part, default)
    else:
      cur = getattr(cur, part, default)
  return cur


def _optional_bool(value, default=False):
  if value in (None, ''):
    return bool(default)
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    return bool(value)
  text = str(value).strip().lower()
  if text in ('1', 'true', 'yes', 'y', 'on'):
    return True
  if text in ('0', 'false', 'no', 'n', 'off'):
    return False
  raise ValueError(f'Cannot parse boolean value: {value!r}')


def _sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
      h.update(chunk)
  return h.hexdigest()


def _resolve_path(path, root):
  path = Path(str(path)).expanduser()
  if path.is_absolute():
    return path
  return Path(root).expanduser().resolve() / path


def _read_yaml(path):
  return yaml.YAML(typ='safe').load(Path(path).read_text())


def _date_int(value):
  text = str(value).strip()
  if not text:
    raise ValueError('empty timestamp')
  # Accept "YYYY-MM-DD HH:MM:SS" and "YYYY/MM/DD HH:MM:SS".
  date = text.split()[0].replace('/', '-')
  parts = date.split('-')
  if len(parts) >= 3:
    return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
  digits = ''.join(ch for ch in text if ch.isdigit())
  if len(digits) < 8:
    raise ValueError(f'Cannot parse date from timestamp: {value!r}')
  return int(digits[:8])


def _csv_date_range(path):
  path = Path(path)
  with path.open(newline='') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    first = next(reader, None)
    if header is None or first is None:
      raise WalkForwardGuardError(f'Training data CSV is empty: {path}')
  last = None
  with path.open('rb') as f:
    f.seek(0, 2)
    pos = f.tell()
    buf = bytearray()
    while pos > 0:
      pos -= 1
      f.seek(pos)
      ch = f.read(1)
      if ch == b'\n':
        if buf:
          break
      else:
        buf.extend(ch)
    if buf:
      last_line = bytes(reversed(buf)).decode('utf-8').strip()
      if last_line:
        last = next(csv.reader([last_line]))
  if last is None:
    with path.open(newline='') as f:
      rows = [row for row in csv.reader(f) if row]
      if len(rows) < 2:
        raise WalkForwardGuardError(f'Training data CSV is empty: {path}')
      last = rows[-1]
  return {
      'path': str(path),
      'min_day': _date_int(first[0]),
      'max_day': _date_int(last[0]),
      'first_timestamp': first[0],
      'last_timestamp': last[0],
  }


def _resolve_market_csv(env_config, *, dreamer_root, data_root):
  trading = env_config.get('trading', {})
  data_path = trading.get('data_path')
  data_interval = trading.get('data_interval', '1m')
  if not data_path:
    raise WalkForwardGuardError('training env config missing trading.data_path')
  candidates = []
  raw = Path(str(data_path)).expanduser()
  if raw.suffix.lower() == '.csv':
    candidates.append(raw if raw.is_absolute() else Path(dreamer_root) / raw)
  candidates.append(Path(data_root) / f'{data_path}_{data_interval}.csv')
  candidates.append(Path(data_root) / f'{data_path}.csv')
  candidates.append(Path(dreamer_root) / 'data' / f'{data_path}_{data_interval}.csv')
  candidates.append(Path(dreamer_root) / 'data' / f'{data_path}.csv')
  for path in candidates:
    if path.exists():
      return path.resolve()
  raise WalkForwardGuardError(
      'Cannot resolve training market CSV for '
      f'data_path={data_path!r}, data_interval={data_interval!r}; '
      f'checked: {[str(x) for x in candidates]}')


def _split_roles(split_manifest):
  if not split_manifest:
    raise WalkForwardGuardError('split_manifest.json is empty')
  final = split_manifest[-1]
  roles = {}
  for key, role in (
      ('train_days', 'train'),
      ('validation_days', 'validation'),
      ('test_days', 'test'),
  ):
    days = [int(x) for x in final.get(key, [])]
    if not days:
      raise WalkForwardGuardError(f'final split missing {key}')
    roles[role] = sorted(days)
  return final.get('name', 'last_fold'), roles


def _range(days):
  return {'min_day': int(min(days)), 'max_day': int(max(days)), 'days': int(len(days))}


def _overlap_days(train_range, days):
  lo, hi = int(train_range['min_day']), int(train_range['max_day'])
  return [int(day) for day in days if lo <= int(day) <= hi]


def validate_walk_forward_split(config, *, dreamer_root=None):
  """Validate the trading walk-forward split used by a formal training run.

  The guard is intentionally metadata-only: it does not construct envs, train,
  evaluate, or change runtime trading behavior.
  """
  guard_cfg = _get(config, 'walk_forward_guard', elements.Config({}))
  enabled = _optional_bool(_get(guard_cfg, 'enabled', True), True)
  audit_version = str(_get(config, 'audit.entry_eval_version', '') or '')
  entry_eval_config_raw = str(_get(config, 'audit.entry_eval_config', '') or '')
  expected_hash = str(_get(config, 'audit.split_manifest_hash', '') or '')
  if not enabled:
    return {'status': 'disabled'}
  if not audit_version and not entry_eval_config_raw and not expected_hash:
    return {'status': 'skipped', 'reason': 'audit entry_eval metadata is empty'}
  if str(_get(config, 'script', 'train')) != 'train':
    return {'status': 'skipped', 'reason': 'guard only enforces script=train'}
  if not entry_eval_config_raw:
    raise WalkForwardGuardError('audit.entry_eval_config is required for formal walk-forward training')
  if not expected_hash:
    raise WalkForwardGuardError('audit.split_manifest_hash is required for formal walk-forward training')

  dreamer_root = Path(dreamer_root or Path(__file__).resolve().parents[1]).resolve()
  data_root = _resolve_path(
      _get(guard_cfg, 'data_root', '/data/logdir/trading_contracts'),
      dreamer_root)
  entry_eval_config = _resolve_path(entry_eval_config_raw, dreamer_root)
  if not entry_eval_config.exists():
    raise WalkForwardGuardError(f'Missing entry-eval config: {entry_eval_config}')
  entry_eval_raw = _read_yaml(entry_eval_config) or {}
  split_path_raw = (
      (entry_eval_raw.get('walk_forward', {}) or {}).get('split_manifest_path', ''))
  if not split_path_raw:
    raise WalkForwardGuardError(
        f'entry-eval config missing walk_forward.split_manifest_path: {entry_eval_config}')
  split_path = _resolve_path(split_path_raw, entry_eval_config.parent)
  if not split_path.exists():
    raise WalkForwardGuardError(f'Missing entry-eval split manifest: {split_path}')

  actual_hash = _sha256(split_path)
  if actual_hash != expected_hash:
    raise WalkForwardGuardError(
        'split_manifest_hash mismatch: '
        f'configured={expected_hash}, actual={actual_hash}, path={split_path}')

  split_manifest = json.loads(split_path.read_text())
  fold_name, roles = _split_roles(split_manifest)

  env_config_path = _resolve_path(
      _get(config, 'env.gymnasium.config_path', ''),
      dreamer_root)
  if not env_config_path.exists():
    raise WalkForwardGuardError(f'Missing training env config: {env_config_path}')
  env_config = _read_yaml(env_config_path)
  market_csv = _resolve_market_csv(env_config, dreamer_root=dreamer_root, data_root=data_root)
  train_data = _csv_date_range(market_csv)

  train_role = _range(roles['train'])
  validation_role = _range(roles['validation'])
  test_role = _range(roles['test'])
  val_overlap = _overlap_days(train_data, roles['validation'])
  test_overlap = _overlap_days(train_data, roles['test'])
  min_before_split_train = int(train_data['min_day']) < int(train_role['min_day'])
  max_exceeds_split_train = int(train_data['max_day']) > int(train_role['max_day'])

  execution_timing = str(_get(config, 'audit.execution_timing', '') or '')
  manifest_timing = (
      (entry_eval_raw.get('entry_evaluator', {}) or {}).get('execution_timing'))
  if execution_timing and manifest_timing and execution_timing != manifest_timing:
    raise WalkForwardGuardError(
        'execution_timing mismatch: '
        f'configured={execution_timing}, entry_eval_config={manifest_timing}')
  manifest_timing = str(manifest_timing or '')

  result = {
      'status': 'pass',
      'role_map_source': 'split_manifest.json:last_fold',
      'entry_eval_version': audit_version,
      'entry_eval_config': str(entry_eval_config),
      'split_manifest_path': str(split_path),
      'split_manifest_hash': actual_hash,
      'split_fold_name': fold_name,
      'env_config_path': str(env_config_path),
      'training_market_csv': str(market_csv),
      'training_data': train_data,
      'roles': {
          'train': train_role,
          'validation': validation_role,
          'test': test_role,
      },
      'execution_timing': execution_timing,
      'manifest_execution_timing': manifest_timing,
      'validation_overlap_days': val_overlap[:20] or [0],
      'validation_overlap_count': len(val_overlap),
      'test_overlap_days': test_overlap[:20] or [0],
      'test_overlap_count': len(test_overlap),
      'training_min_before_split_train': min_before_split_train,
      'training_max_exceeds_split_train': max_exceeds_split_train,
  }
  problems = []
  if min_before_split_train:
    problems.append(
        'training data starts before split train min day '
        f'({train_data["min_day"]} < {train_role["min_day"]})')
  if max_exceeds_split_train:
    problems.append(
        'training data extends beyond split train max day '
        f'({train_data["max_day"]} > {train_role["max_day"]})')
  if val_overlap:
    problems.append(f'training data overlaps validation days ({len(val_overlap)} days)')
  if test_overlap:
    problems.append(f'training data overlaps test days ({len(test_overlap)} days)')
  if problems:
    result['status'] = 'non_holdout'
    result['problems'] = problems
    if not _optional_bool(_get(guard_cfg, 'allow_non_holdout_training', False), False):
      raise WalkForwardGuardError('; '.join(problems))
  return result
