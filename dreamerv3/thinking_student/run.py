"""Thinking Student V1 runner: autonomous Native rollout, L1 training, resume.

  python -m dreamerv3.thinking_student.run rollout --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run train-market --run DIR --corpus DIR ...

A run directory holds `ckpt/<step>/agent.pkl` and a `latest` pointer. Loading
never checks code or data hashes (C2); shape mismatches fail loudly instead
of silently re-initializing.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .config import observation_space
from .student import make_agent

POLICY_OBS = tuple(observation_space())


# ---- checkpoint ----

def save(agent, run, tag=None):
  run = Path(run)
  data = agent.save()
  name = tag or f"{data['counters']['updates']:012d}"
  folder = run / 'ckpt' / name
  folder.mkdir(parents=True, exist_ok=True)
  tmp = folder / 'agent.pkl.tmp'
  with tmp.open('wb') as f:
    pickle.dump(data, f)
  tmp.rename(folder / 'agent.pkl')
  (run / 'ckpt' / 'latest').write_text(name)
  return folder


def latest(run):
  pointer = Path(run) / 'ckpt' / 'latest'
  return Path(run) / 'ckpt' / pointer.read_text().strip() if pointer.exists() else None


def resume(agent, run):
  folder = latest(run)
  if folder is None:
    return None
  with (folder / 'agent.pkl').open('rb') as f:
    agent.load(pickle.load(f))
  return folder


def build(run, preset, device, init_f1=None, **kwargs):
  agent = make_agent(Path(run) / 'logdir', preset=preset, device=device, **kwargs)
  folder = resume(agent, run)
  if folder is not None:
    print(f'Resumed {folder}')
  elif init_f1:
    from .transfer import inherit_f1
    inherit_f1(agent, init_f1)
  return agent


# ---- autonomous Native rollout through the single student.policy ----

def student_obs(obs, reward, first, last, terminal):
  out = {k: np.asarray(obs[k])[None] for k in POLICY_OBS
         if k not in ('reward', 'is_first', 'is_last', 'is_terminal')}
  out.update(reward=np.float32([reward]), is_first=np.array([first]),
             is_last=np.array([last]), is_terminal=np.array([terminal]))
  return out


def rollout(agent, env, steps, mode='train', carry=None, record=None, seed=0,
            options=None):
  """Student owns the account: every executed action is the student's own."""
  obs, info = env.reset(seed=seed, options=options)
  carry = agent.init_policy(1) if carry is None else carry
  reward, first, rows = 0., True, []
  for step in range(steps):
    o = student_obs(obs, reward, first, False, False)
    carry, act, out = agent.policy(carry, o, mode=mode)
    action = int(act['action'][0])
    mask = obs['action_mask'].astype(float).tolist()
    if action == 1 and mask[1] < 0.5:
      raise RuntimeError('Student policy proposed a masked ADD')
    obs, reward, terminated, truncated, info = env.step(action)
    row = dict(step=step, cursor=int(env.current_step), mask=mask,
               proposed=action, executed=action, greedy=bool(out['greedy'][0]),
               reward=float(reward), status=info['status'],
               reject_reason=info.get('reject_reason'),
               **{k.split('/', 1)[1]: float(v[0]) for k, v in out.items()
                  if k.startswith('answer/')})
    rows.append(row)
    if record is not None:
      record.write(json.dumps(row) + '\n')
    first = False
    if info['status'] == 'CHUNK_END':
      obs, info = env.reset()
    elif terminated or truncated:
      break
  return carry, rows


# ---- L1 market/account training on natural corpus sequences ----

class CorpusWindows:
  """Random T-step windows inside one replay chunk of the P0-P4 corpus."""

  def __init__(self, corpus, policies=None, seed=0):
    root = Path(corpus) / 'policies'
    names = policies or sorted(p.name for p in root.iterdir() if p.is_dir())
    self.files = [f for n in names for f in sorted((root / n / 'replay').glob('*.npz'))]
    if not self.files:
      raise FileNotFoundError(f'No replay chunks under {root}')
    self.rng = np.random.default_rng(seed)
    self.cache = {}

  def _chunk(self, path):
    if path not in self.cache:
      if len(self.cache) > 16:
        self.cache.pop(next(iter(self.cache)))
      with np.load(path) as d:
        self.cache[path] = {k: d[k] for k in d.files}
    return self.cache[path]

  def sample(self, batch, length):
    rows = []
    for _ in range(batch):
      chunk = self._chunk(self.files[self.rng.integers(len(self.files))])
      n = len(chunk['reward'])
      start = int(self.rng.integers(0, n - length + 1))
      rows.append({k: chunk[k][start:start + length]
                   for k in (*POLICY_OBS, 'action') if k in chunk})
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    data['is_first'] = data['is_first'].copy()
    data['is_first'][:, 0] = True
    # Corpus predates account_native; mark it invalid instead of inventing it.
    data['account_native'] = np.zeros((batch, length, 10), np.float32)
    data['account_valid'] = np.zeros((batch, length), np.float32)
    return data


def train_market(agent, windows, updates, log_every=10):
  from embodied.jax import internal
  mets = {}
  for i in range(updates):
    data = windows.sample(agent.config.batch_size, agent.config.batch_length)
    data = internal.device_put(data, agent.train_sharded)
    data['seed'] = agent._seeds(int(agent.n_batches), agent.train_mirrored)
    agent.n_batches.increment()
    carry = agent.init_train(agent.config.batch_size)
    _, _, mets = agent.train(carry, data)
    if (i + 1) % log_every == 0 or i == updates - 1:
      keys = ('opt_l1/loss', 'loss/dyn', 'loss/market_seq_e_shared',
              'loss/account_can_add', 'brain/p_add', 'brain/delta_q_abs')
      print(f'update {int(agent.n_updates)}', {
          k: round(float(mets[k]), 4) for k in keys if k in mets})
  return mets


def corpus_env(corpus):
  """Native E env on the corpus TRAIN source, starting at its warm account."""
  from e_data_v1.provider import ETrainAdmission
  from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
  from gym_trading_env.research.ts_v1_account import StudentEnv
  corpus = Path(corpus)
  lock = json.loads((corpus / 'source_lock.json').read_text())
  profile = ETrainAdmission.from_train_pool(
      lock['source_pool']['path'], corpus / 'e_source',
      corpus / 'selected_train_pool.json', contracts=[lock['contract']])
  config = corpus / 'inputs' / lock['contract'] / 'env.yaml'
  initial = json.loads((corpus / 'initial_account.json').read_text())
  env = StudentEnv(FixedDirectionEntryEnv(config_path=config, e_profile=profile))
  return env, {'resume': initial}


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('command', choices=['rollout', 'train-market'])
  parser.add_argument('--run', required=True)
  parser.add_argument('--preset', default='size12m')
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--init-f1', default=None)
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--mode', default='train', choices=['train', 'eval'])
  parser.add_argument('--corpus', default=None)
  parser.add_argument('--updates', type=int, default=10)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--batch-length', type=int, default=64)
  parser.add_argument('--fixture', default=None,
                      help='synthetic native env folder (engineering smoke only)')
  args = parser.parse_args(argv)
  agent = build(args.run, args.preset, args.device, args.init_f1,
                batch_size=args.batch_size, batch_length=args.batch_length)
  if args.command == 'train-market':
    train_market(agent, CorpusWindows(args.corpus), args.updates)
    print('Saved', save(agent, args.run))
  else:
    options = None
    if args.fixture:
      from wp02_fixture import native_factory
      from gym_trading_env.research.ts_v1_account import StudentEnv
      factory, _, _ = native_factory(args.fixture, minutes=210, chunk=64)
      env = StudentEnv(factory())
    elif args.corpus:
      env, options = corpus_env(args.corpus)
    else:
      raise ValueError('rollout needs --fixture or --corpus')
    out = Path(args.run) / 'rollouts'
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'rollout_{int(agent.n_actions):09d}.jsonl').open('w') as f:
      _, rows = rollout(agent, env, args.steps, args.mode, record=f,
                        options=options)
    adds = sum(r['executed'] == 1 for r in rows)
    print(f'rollout steps={len(rows)} adds={adds} '
          f'reward={sum(r["reward"] for r in rows):.3f}')
    if not latest(args.run):
      print('Saved', save(agent, args.run))


if __name__ == '__main__':
  main()
