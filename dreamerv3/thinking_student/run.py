"""Thinking Student V1 runner: autonomous Native rollout, staged training, resume.

  python -m dreamerv3.thinking_student.run rollout --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run train-market --run DIR --corpus DIR ...
  python -m dreamerv3.thinking_student.run train-outcome --run DIR --corpus DIR ...

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
from .corpus import CorpusWindows
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


def build(run, preset, device, init_f1=None, init_from=None, **kwargs):
  """Resume the run strictly; a fresh run may start from F1 market weights
  or from another Thinking Student checkpoint (`init_from`, listed changes)."""
  if init_f1 and init_from:
    raise ValueError('Choose one of init_f1 and init_from')
  agent = make_agent(Path(run) / 'logdir', preset=preset, device=device, **kwargs)
  folder = resume(agent, run)
  if folder is not None:
    print(f'Resumed {folder}')
  elif init_from:
    from .transfer import inherit_student
    lists = inherit_student(agent, init_from)
    Path(run).mkdir(parents=True, exist_ok=True)
    (Path(run) / 'inherit.json').write_text(json.dumps(lists, indent=1))
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


# ---- staged training on natural corpus sequences ----

LOG_KEYS = {
    'L1': ('opt_l1/loss', 'loss/dyn', 'loss/market_seq_e_shared',
           'loss/account_can_add', 'brain/p_add', 'brain/delta_q_abs'),
    'L2': ('opt_l2/loss', 'loss/outcome_event', 'loss/outcome_risk',
           'loss/outcome_mae', 'loss/outcome_quality', 'outcome/labels',
           'outcome/p_tp_on_tp', 'outcome/p_tp_on_sl'),
}


def _device(agent, data, counter):
  from embodied.jax import internal
  data = internal.device_put(data, agent.train_sharded)
  data['seed'] = agent._seeds(counter, agent.train_mirrored)
  return data


def train_stage(agent, windows, updates, log_every=10, on_update=None):
  """`updates` optimizer steps of the configured A6 stage on corpus windows."""
  stage, mets = agent.config.ts.stage, {}
  for i in range(updates):
    data = windows.sample(agent.config.batch_size, agent.config.batch_length)
    data = _device(agent, data, int(agent.n_batches))
    agent.n_batches.increment()
    _, _, mets = agent.train(agent.init_train(agent.config.batch_size), data)
    if (i + 1) % log_every == 0 or i == updates - 1:
      print(f'update {int(agent.n_updates)}', {
          k: round(float(mets[k]), 4) for k in LOG_KEYS[stage] if k in mets})
    if on_update:
      on_update(i + 1)
  return mets


# Denominator of each pooled L2 metric; the rest average over labelled rows.
POOL = {'loss/outcome_risk': 'outcome/n_risk', 'loss/outcome_mae': 'outcome/n_mae',
        'loss/outcome_quality': 'outcome/n_quality',
        'outcome/p_tp_on_tp': 'outcome/n_tp', 'outcome/p_tp_on_sl': 'outcome/n_sl'}


def evaluate(agent, windows, constant=None):
  """L2 report pooled over every held-out window. `constant` (a class
  distribution) is scored on exactly the rows the Outcome Brain is scored on."""
  from .student import warm
  B, T = agent.config.batch_size, agent.config.batch_length
  sums, weights, batches = {}, {}, 0

  def pool(key, value, weight):
    sums[key] = sums.get(key, 0.) + weight * float(value)
    weights[key] = weights.get(key, 0.) + weight

  for i, data in enumerate(windows.batches(B, T)):
    if constant is not None:
      w = np.asarray(warm(data['is_first'], agent.config.ts.outcome_burnin))
      w = w * data['outcome_valid']
      nll = -np.log((data['outcome_allowed'] * constant).sum(-1) + (w == 0))
      pool('constant/outcome_event', (nll * w).sum() / max(w.sum(), 1.), w.sum())
    _, mets = agent.report(agent.init_report(B), _device(agent, data, i))
    batches += 1
    for k, v in mets.items():
      if not k.startswith(('loss/', 'outcome/')) or k.startswith('outcome/n_'):
        continue
      if k in ('outcome/labels', 'outcome/label_not_legal'):
        sums[k] = sums.get(k, 0.) + float(v)
      else:
        pool(k, v, float(mets[POOL.get(k, 'outcome/labels')]))
  out = {k: round(v / max(weights[k], 1.), 5) if k in weights else int(v)
         for k, v in sorted(sums.items())}
  return {'updates': int(agent.n_updates), 'batches': batches, **out}


def constant_answer(train):
  """Add-one smoothed class frequencies of the observed train outcomes: the
  best Outcome answer that ignores the state, as a reference to beat."""
  fit = np.concatenate([
      train.labels[name]['outcome_allowed'][lo:hi][
          train.labels[name]['outcome_valid'][lo:hi] > 0]
      for name, _, lo, hi in train.chunks])
  observed = fit[fit.sum(-1) == 1]
  return (observed.sum(0) + 1.) / (len(observed) + fit.shape[-1])


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
  parser.add_argument('command', choices=['rollout', 'train-market', 'train-outcome'])
  parser.add_argument('--run', required=True)
  parser.add_argument('--preset', default='size12m')
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--init-f1', default=None)
  parser.add_argument('--init-from', default=None,
                      help='earlier Thinking Student run or checkpoint (listed inheritance)')
  parser.add_argument('--label-cache', default=None)
  parser.add_argument('--log-every', type=int, default=10)
  parser.add_argument('--eval-every', type=int, default=0)
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--mode', default='train', choices=['train', 'eval'])
  parser.add_argument('--corpus', default=None)
  parser.add_argument('--updates', type=int, default=10)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--batch-length', type=int, default=64)
  parser.add_argument('--fixture', default=None,
                      help='synthetic native env folder (engineering smoke only)')
  args = parser.parse_args(argv)
  stage = {'train-market': 'L1', 'train-outcome': 'L2'}.get(args.command, 'L1')
  agent = build(args.run, args.preset, args.device, args.init_f1, args.init_from,
                batch_size=args.batch_size, batch_length=args.batch_length,
                updates={'agent.ts.stage': stage})
  if args.command == 'train-market':
    windows = CorpusWindows(args.corpus, labels=False)
    train_stage(agent, windows, args.updates, args.log_every)
    print('Saved', save(agent, args.run))
  elif args.command == 'train-outcome':
    cache = args.label_cache or Path(args.run) / 'label_cache'
    train = CorpusWindows(args.corpus, cache=cache, split='train')
    held = CorpusWindows(args.corpus, cache=cache, split='eval')
    constant = constant_answer(train)
    report = {'train_chunks': len(train.chunks), 'eval_chunks': len(held.chunks),
              'constant_answer': constant.round(5).tolist(), 'eval': []}

    def checkpoint(done=None):
      if done is None or (args.eval_every and done % args.eval_every == 0
                          and done != args.updates):
        report['eval'].append(evaluate(agent, held, constant))
        print('eval', report['eval'][-1])

    checkpoint()
    report['last_train'] = {k: float(v) for k, v in train_stage(
        agent, train, args.updates, args.log_every, checkpoint).items()
        if k.startswith(LOG_KEYS['L2'])}
    checkpoint()
    folder = save(agent, args.run)
    (folder / 'train_outcome.json').write_text(json.dumps(report, indent=1))
    print('Saved', folder)
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
