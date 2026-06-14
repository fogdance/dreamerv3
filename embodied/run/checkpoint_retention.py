import json
import pathlib
import shutil
import time


class StepCheckpointRetention:

  def __init__(self, logdir, config):
    self._logdir = pathlib.Path(logdir)
    self._config = config or {}
    self._enabled = bool(self._get('enabled', False))
    self._targets = (
        self._parse_steps(self._get('steps', [])) if self._enabled else [])
    directory = self._get('directory', 'ckpt_retained')
    self._directory = self._logdir / str(directory)
    self._manifest_path = self._directory / 'retention_manifest.json'
    self._manifest = self._load_manifest()

  def maybe_retain(self, checkpoint, step):
    if not self.has_due(step):
      return
    current = int(step)
    due = self._due_targets(current)
    latest = checkpoint.latest()
    if latest is None:
      raise FileNotFoundError('Cannot retain checkpoint because latest is missing')
    latest = pathlib.Path(latest)
    if not latest.is_dir():
      raise FileNotFoundError(f'Cannot retain missing checkpoint: {latest}')
    self._directory.mkdir(parents=True, exist_ok=True)
    for target in due:
      self._retain_target(target, current, latest)
    self._write_manifest()

  def has_due(self, step):
    return bool(self._due_targets(int(step)))

  def _due_targets(self, current):
    if not self._enabled or not self._targets:
      return []
    return [
        target for target in self._targets
        if current >= target and not self._has_target(target)]

  def _retain_target(self, target, current, source):
    name = f'step_{target:012d}'
    dest = self._directory / name
    if dest.exists():
      status = 'already_exists'
    else:
      tmp = self._directory / f'.{name}.tmp'
      if tmp.exists():
        shutil.rmtree(tmp)
      shutil.copytree(source, tmp)
      tmp.rename(dest)
      status = 'copied'
    self._manifest['targets'][str(target)] = {
        'target_step': target,
        'retained_at_step': current,
        'source': str(source),
        'path': str(dest),
        'status': status,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    print(
        f'[checkpoint_retention] retained target={target} '
        f'at step={current}: {dest}')

  def _has_target(self, target):
    record = self._manifest.get('targets', {}).get(str(target))
    if not record:
      return False
    path = pathlib.Path(record.get('path', ''))
    return path.is_dir()

  def _load_manifest(self):
    if self._manifest_path.exists():
      try:
        return json.loads(self._manifest_path.read_text())
      except Exception:
        pass
    return {
        'enabled': self._enabled,
        'directory': str(self._directory),
        'configured_steps': self._targets,
        'targets': {},
    }

  def _write_manifest(self):
    self._manifest.update({
        'enabled': self._enabled,
        'directory': str(self._directory),
        'configured_steps': self._targets,
    })
    self._manifest_path.write_text(json.dumps(
        self._manifest, ensure_ascii=False, indent=2))

  def _get(self, key, default=None):
    if hasattr(self._config, 'get'):
      return self._config.get(key, default)
    return getattr(self._config, key, default)

  @staticmethod
  def _parse_steps(value):
    if value in (None, '', False):
      return []
    if isinstance(value, str):
      value = [x.strip() for x in value.split(',') if x.strip()]
    if not isinstance(value, (list, tuple)):
      value = [value]
    steps = []
    for item in value:
      step = int(float(item))
      if step <= 0:
        raise ValueError(f'Checkpoint retention step must be positive: {item}')
      steps.append(step)
    return sorted(set(steps))
