"""Evidence for the opt-in continuous Single3 offline experiment."""
import copy
import json
import math
import os
import time
import traceback
import uuid
from pathlib import Path

import numpy as np


def atomic_json(path, value):
  path = Path(path)
  temporary = path.with_name(path.name + '.pending.' + uuid.uuid4().hex)
  temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
  os.replace(temporary, path)


def failure_evidence(logdir, error):
  root = Path(logdir)
  if root.is_dir():
    atomic_json(root / ('failure_' + uuid.uuid4().hex + '.json'), dict(
        status='FAILED', time=time.time(), error=repr(error),
        traceback=traceback.format_exc(), training_complete=False))


class TrainingAudit:
  """Authoritative counters are checkpointed; logs retain all failed attempts."""
  def __init__(self, directory, batch_positions):
    self.directory = Path(directory)
    self.batch_positions = int(batch_positions)
    self.attempt = uuid.uuid4().hex
    self.state = dict(last_update=0, training_positions=0, gate_open_updates=0,
        first_gate_open_update=None, actor_effective_updates=0,
        critic_effective_updates=0, reward_effective_updates=0,
        wm_effective_updates=0, actor_changes_gate_closed=0)

  def observe(self, update, metrics):
    update = int(update)
    if update != self.state['last_update'] + 1 or int(metrics['train/update_index']) != update:
      raise ValueError('Single3 training metrics/update counter mismatch')
    required = ['opt/finite', 'opt/loss', 'opt/grad_norm', 'avail/ready']
    for head in ('actor', 'critic', 'reward', 'wm'):
      required += ['opt/' + head + suffix for suffix in ('_grad_norm', '_delta_norm', '_changed')]
    if any(k not in metrics for k in required):
      raise ValueError('Missing real neural update diagnostics')
    if not bool(metrics['opt/finite']) or any(
        not np.isfinite(np.asarray(metrics[k])).all() for k in required):
      raise FloatingPointError('Non-finite loss, gradient, parameter, update or optimizer state')
    opened = bool(metrics['avail/ready'])
    scalars, unavailable = {}, []
    for key, value in metrics.items():
      array = np.asarray(value)
      if array.size == 1:
        number = float(array.reshape(()))
        scalars[key] = number if math.isfinite(number) else None
        if not math.isfinite(number):
          unavailable.append(key)
    with (self.directory / 'train_updates.jsonl').open('a') as stream:
      stream.write(json.dumps(dict(attempt=self.attempt, update=update,
          time=time.time(), gate_used=opened, metrics=scalars,
          unavailable_metrics=unavailable), allow_nan=False) + '\n')
    self.state['last_update'] = update
    self.state['training_positions'] += self.batch_positions
    self.state['gate_open_updates'] += int(opened)
    if opened and self.state['first_gate_open_update'] is None:
      self.state['first_gate_open_update'] = update
    for head in ('actor', 'critic', 'reward', 'wm'):
      changed = bool(metrics['opt/' + head + '_changed'])
      has_gradient = float(metrics['opt/' + head + '_grad_norm']) > 0
      enabled = opened or head in ('reward', 'wm')
      self.state[head + '_effective_updates'] += int(enabled and changed and has_gradient)
      if head == 'actor' and not opened:
        self.state['actor_changes_gate_closed'] += int(changed)

  def save(self):
    return copy.deepcopy(self.state)

  def load(self, state):
    if set(state) != set(self.state):
      raise ValueError('Training audit checkpoint schema mismatch')
    self.state = copy.deepcopy(state)

  def summary(self, identity, checkpoint, budget, *, experiment='C1_M1_1M'):
    if self.state['last_update'] != budget:
      raise ValueError('Single3 training budget not completed')
    return dict(status='TRAINING_COMPLETE', experiment=experiment,
        identity_hash=identity, checkpoint=str(checkpoint), **self.save(),
        actor_training='EFFECTIVE_UPDATES_OBSERVED' if self.state['actor_effective_updates'] else 'NO_EFFECTIVE_UPDATES',
        timing_status='NOT_ESTABLISHED', evaluation_status='NOT_RUN')
