"""Opt-in FG full-state checkpoints; legacy Dreamer checkpoints are unchanged."""
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import sys
import uuid
from pathlib import Path

import cloudpickle
import elements
import numpy as np


def host(value):
  if isinstance(value, dict):
    return {k: host(v) for k, v in value.items()}
  if isinstance(value, tuple):
    return tuple(host(v) for v in value)
  if isinstance(value, list):
    return [host(v) for v in value]
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    if not isinstance(value, np.ndarray):
      import jax
      value = jax.device_get(value)
    return np.array(value, copy=True)
  return copy.deepcopy(value)


def plain(value):
  if hasattr(value, 'items'):
    return {k: plain(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [plain(v) for v in value]
  return value


def digest(value):
  return hashlib.sha256(json.dumps(plain(value), sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
      h.update(block)
  return h.hexdigest()


def learning_spec(config):
  c = plain(config)
  keys = ('agent', 'jax', 'replay', 'seed', 'experiment_seed', 'seed_protocol',
      'batch_size', 'batch_length', 'report_length', 'replay_context',
      'consec_train', 'consec_report', 'action_mask_warmup')
  result = {k: c[k] for k in keys if k in c}
  result['run'] = {k: c['run'][k] for k in (
      'train_ratio', 'envs', 'env_checkpoint', 'report_batches', 'debug',
      'fg_full_checkpoint', 'fg_report_every_steps', 'fg_save_every_steps')
      if k in c['run']}
  return result


def provenance(config):
  c = plain(config)
  from gym_trading_env.envs.entry_contract import fingerprint
  manifest = json.loads(Path(c['fg_entry']['manifest']).read_text())
  expected = manifest.pop('content_hash')
  if fingerprint(manifest) != expected:
    raise ValueError('FG checkpoint dataset manifest changed')
  versions = {}
  for package in ('numpy', 'jax', 'jaxlib', 'optax', 'elements', 'portal', 'cloudpickle'):
    versions[package] = importlib.metadata.version(package)
  import gym_trading_env
  roots = [Path(__file__).resolve().parents[2], Path(gym_trading_env.__file__).resolve().parent]
  code = {}
  for root in roots:
    for name in ('embodied', 'dreamerv3') if root == roots[0] else ('envs', 'utils'):
      for path in sorted((root / name).rglob('*.py')):
        if 'tests' not in path.parts:
          code[f'{root.name}/{path.relative_to(root)}'] = file_hash(path)
  return dict(schema=1, learning=learning_spec(c), dataset=expected,
      selector_hash=file_hash(c['env']['gymnasium']['config_path']),
      env_seed=c['env']['train_seed'], code_hash=digest(code),
      python=sys.version, versions=versions)


class StepSchedule:
  def __init__(self, report_every, save_every, key_steps=()):
    self.report_every, self.save_every = int(report_every), int(save_every)
    if min(self.report_every, self.save_every) <= 0:
      raise ValueError('FG fixed step intervals must be positive')
    self.key_steps = sorted(set(int(x) for x in key_steps))
    self.next_report, self.next_save = self.report_every, self.save_every

  def advance(self, step):
    step = int(step)
    report, save = step == self.next_report, step == self.next_save
    if step > min(self.next_report, self.next_save):
      raise ValueError('FG fixed-step boundary overshot')
    if report:
      self.next_report += self.report_every
    if save:
      self.next_save += self.save_every
    return report, save or step in self.key_steps

  def distance(self, step, budget):
    boundaries = [int(budget), self.next_report, self.next_save]
    boundaries += [s for s in self.key_steps if s > int(step)]
    return min(boundaries) - int(step)

  def save(self):
    return dict(next_report=self.next_report, next_save=self.next_save)

  def load(self, data):
    self.next_report, self.next_save = data['next_report'], data['next_save']


def save_replay(replay, directory):
  from embodied.core.selectors import Uniform
  if not isinstance(replay.sampler, Uniform):
    raise ValueError('FG full checkpoints currently require Uniform Replay')
  directory.mkdir()
  with replay.rwlock.writing, replay.refs_lock, replay.sampler.lock:
    chunks = []
    for chunk in replay.chunks.values():
      name = str(chunk.uuid) + '.npz'
      arrays = {} if chunk.data is None else {k: v[:chunk.length] for k, v in chunk.data.items()}
      np.savez_compressed(directory / name, **arrays)
      chunks.append(dict(uuid=chunk.uuid, succ=chunk.succ, time=chunk.time,
          size=chunk.size, length=chunk.length, name=name))
    fields = ('refs', 'items', 'fifo', 'itemid', 'current', 'streams', 'metrics')
    state = {k: copy.deepcopy(getattr(replay, k)) for k in fields}
    if replay.online:
      state.update(lengths=copy.deepcopy(replay.lengths), queue=copy.deepcopy(replay.queue))
    state.update(chunks=chunks, length=replay.length, capacity=replay.capacity,
        chunksize=replay.chunksize, online=replay.online,
        sampler=dict(indices=replay.sampler.indices.copy(), keys=replay.sampler.keys.copy(),
            rng=copy.deepcopy(replay.sampler.rng.bit_generator.state)))
    return state


def load_replay(replay, state, directory):
  from embodied.core.chunk import Chunk
  from embodied.core.selectors import Uniform
  for key in ('length', 'capacity', 'chunksize', 'online'):
    if getattr(replay, key) != state[key]:
      raise ValueError(f'FG Replay structure mismatch: {key}')
  if not isinstance(replay.sampler, Uniform):
    raise ValueError('FG full checkpoints require Uniform Replay')
  chunks = {}
  for meta in state['chunks']:
    chunk = Chunk.__new__(Chunk)
    for key in ('uuid', 'succ', 'time', 'size', 'length'):
      setattr(chunk, key, meta[key])
    with np.load(directory / meta['name'], allow_pickle=False) as data:
      chunk.data = {}
      for key in data.files:
        value = data[key]
        array = np.empty((chunk.size, *value.shape[1:]), value.dtype)
        array[:chunk.length] = value
        chunk.data[key] = array
      chunk.data = chunk.data or None
    chunk.saved = False
    chunks[chunk.uuid] = chunk
  replay.chunks = chunks
  for key in ('refs', 'items', 'fifo', 'itemid', 'current', 'streams', 'metrics', 'lengths', 'queue'):
    if key in state:
      setattr(replay, key, copy.deepcopy(state[key]))
  replay.sampler.indices = state['sampler']['indices'].copy()
  replay.sampler.keys = state['sampler']['keys'].copy()
  replay.sampler.rng.bit_generator.state = state['sampler']['rng']
  if replay.directory:
    replay.saved = set()


def validate_bundle(path, expected_identity=None):
  path = Path(path)
  if not (path / 'done').is_file():
    raise ValueError(f'Incomplete FG full checkpoint: {path}')
  meta = json.loads((path / 'manifest.json').read_text())
  if meta.get('schema') != 1 or meta.get('protocol') != 'FG_FULL_V1':
    raise ValueError('Unsupported FG full checkpoint protocol')
  if expected_identity is not None and meta['identity'] != plain(expected_identity):
    raise ValueError('FG full checkpoint identity mismatch')
  for filename, expected in meta['files'].items():
    if file_hash(path / filename) != expected:
      raise ValueError(f'FG full checkpoint corrupt file: {filename}')
  actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
  if actual != set(meta['files']) | {'done', 'manifest.json'}:
    raise ValueError('FG full checkpoint file set mismatch')
  step = pickle.loads((path / 'step.pkl').read_bytes())
  if int(step) != meta['step']:
    raise ValueError('FG full checkpoint step mismatch')
  return meta


class FullCheckpoint(elements.Checkpoint):
  def __init__(self, directory, step, identity, streams, key_steps=(), reserve_gib=10):
    super().__init__(directory, keep=None, step=step)
    self._directory = Path(directory)
    self._identity, self._streams = plain(identity), streams
    self._key_steps = set(int(x) for x in key_steps)
    self._reserve = float(reserve_gib) * 1024 ** 3

  def latest(self):
    if not self._directory.exists():
      return None
    candidates = sorted((p for p in self._directory.glob('step_*') if (p / 'done').is_file()), reverse=True)
    for path in candidates:
      try:
        validate_bundle(path)
      except (ValueError, OSError, KeyError, json.JSONDecodeError, pickle.UnpicklingError):
        continue
      # Never fall back to a different experiment identity.
      validate_bundle(path, self._identity)
      return path
    if any(self._directory.iterdir()):
      raise ValueError('No valid FG full recovery point; refusing fresh start')
    return None

  @contextlib.contextmanager
  def _pause(self):
    with contextlib.ExitStack() as stack:
      for stream in self._streams:
        if hasattr(stream, 'pause'):
          stack.enter_context(stream.pause())
      yield

  def save(self, path=None, keys=None, final=False):
    if path or keys:
      raise ValueError('FG full checkpoint does not allow partial saves')
    self._directory.mkdir(parents=True, exist_ok=True)
    step = int(self._step)
    dest = self._directory / f'step_{step:012d}'
    if dest.exists():
      meta = validate_bundle(dest, self._identity)
      if final or step in self._key_steps:
        if not meta['protected']:
          meta['protected'] = True
          tmp = dest / '.manifest.tmp'
          tmp.write_text(json.dumps(meta, sort_keys=True, indent=2))
          os.replace(tmp, dest / 'manifest.json')
      self._publish(dest)
      return
    replay = self._saveables['replay']
    estimate = sum(x.nbytes for x in replay.chunks.values())
    estimate += sum(getattr(v, 'nbytes', 0) for v in getattr(self._saveables['agent'], 'params', {}).values())
    latest = self.latest()
    if latest:
      estimate += (latest / 'agent.pkl').stat().st_size
    if shutil.disk_usage(self._directory).free < self._reserve + estimate:
      raise OSError('Insufficient space for complete FG snapshot and reserved disk space')
    temp = self._directory / ('.pending_' + uuid.uuid4().hex)
    temp.mkdir()
    with self._pause():
      for name, obj in self._saveables.items():
        state = (save_replay(obj, temp / 'replay') if name == 'replay' else
            obj.save_exact() if name == 'agent' and hasattr(obj, 'save_exact') else obj.save())
        serializer = cloudpickle if name == 'runtime' else pickle
        (temp / f'{name}.pkl').write_bytes(serializer.dumps(state, protocol=5))
      config = self._directory.parent / 'config.yaml'
      if config.exists():
        shutil.copyfile(config, temp / 'config.yaml')
      meta = dict(schema=1, protocol='FG_FULL_V1', step=step, identity=self._identity,
          protected=bool(final or step in self._key_steps),
          files={str(p.relative_to(temp)): file_hash(p) for p in sorted(temp.rglob('*')) if p.is_file()})
      (temp / 'manifest.json').write_text(json.dumps(meta, sort_keys=True, indent=2))
      (temp / 'done').write_text('')
      validate_bundle(temp, self._identity)
      temp.rename(dest)
    self._publish(dest)

  def _publish(self, dest):
    pointer = self._directory / '.latest.tmp'
    pointer.write_text(dest.name)
    os.replace(pointer, self._directory / 'latest')
    regular = []
    for candidate in sorted(self._directory.glob('step_*')):
      try:
        meta = validate_bundle(candidate)
      except (ValueError, OSError, KeyError, json.JSONDecodeError, pickle.UnpicklingError):
        continue
      if meta['identity'] == self._identity and not meta['protected']:
        regular.append(candidate)
    for candidate in regular[:-2]:
      shutil.rmtree(candidate)

  def load(self, path=None, keys=None):
    if keys:
      raise ValueError('FG full checkpoint does not allow partial restores')
    path = Path(path or self.latest())
    validate_bundle(path, self._identity)
    for name, obj in self._saveables.items():
      state = pickle.loads((path / f'{name}.pkl').read_bytes())
      if name == 'replay':
        load_replay(obj, state, path / 'replay')
      elif name == 'agent':
        obj.load_exact(state)
      else:
        obj.load(state)


class RuntimeState:
  def __init__(self, schedule, ratio, streams, get_carry, set_carry, aggregates, samples):
    self.schedule, self.ratio, self.streams = schedule, ratio, streams
    self.get_carry, self.set_carry = get_carry, set_carry
    self.aggregates, self.samples = aggregates, samples
    self.started = False

  def start(self):
    if not self.started:
      for stream in self.streams:
        iter(stream)
      self.started = True

  def save(self):
    return dict(schedule=self.schedule.save(), ratio=copy.deepcopy(self.ratio.__dict__),
        streams=[s.save_full() if hasattr(s, 'save_full') else s.save() for s in self.streams],
        carry=host(self.get_carry()), samples=self.samples[0], started=self.started,
        aggregates={k: copy.deepcopy(v.reducers) for k, v in self.aggregates.items()},
        python_rng=random.getstate(), numpy_rng=np.random.get_state())

  def load(self, data):
    self.schedule.load(data['schedule'])
    self.ratio.__dict__.update(data['ratio'])
    for stream, state in zip(self.streams, data['streams']):
      stream.load_full(state) if hasattr(stream, 'load_full') else stream.load(state)
    self.set_carry(data['carry'])
    self.samples[0] = data['samples']
    for key, reducers in data['aggregates'].items():
      self.aggregates[key].reducers = reducers
    random.setstate(data['python_rng'])
    np.random.set_state(data['numpy_rng'])
    # Start only after all checkpoint members (including Agent) are restored.
    self.resume_started = data['started']
