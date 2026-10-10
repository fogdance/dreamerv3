"""Real training-loop contract with a non-neural asynchronous metric stub."""
from functools import partial
import pickle

import elements
import embodied

from embodied.tests.test_fg_full_checkpoint import CountingAgent, CountingEnv


class AsyncMetricsAgent(CountingAgent):
  def train(self, carry, batch):
    carry, outs, metrics = super().train(carry, batch)
    # Actual jax.Agent.train() returns no metrics on its first update.
    if self.updates == 1:
      return carry, outs, {}
    for prefix in ('post', 'prior'):
      metrics[f'avail/{prefix}_add_positives'] = 1
      metrics[f'avail/{prefix}_add_true_positives'] = 1
    return carry, outs, metrics


def test_fg_training_skips_initial_pending_metrics_and_pools_counts(tmp_path):
  from embodied.run.train import train
  agent = AsyncMetricsAgent()
  warmup = dict(enabled=True, min_train_samples=1, consecutive_reports=1,
      regression_reports=5, stop_on_regression=True, post_accuracy=.995,
      post_exact=.990, prior_accuracy=.990, prior_exact=.980, prior_fpr=.005,
      img_fallback=.0001, add_recall_enabled=True, min_add_positives=2,
      post_add_recall=.8, prior_add_recall=.8)
  args = elements.Config(steps=13, train_ratio=4, log_every=1e9,
      report_every=1e9, save_every=1e9, report_batches=1, from_checkpoint='',
      fg_stage_transfer='', env_checkpoint=True, debug=True,
      usage=dict(psutil=False,nvsmi=False,gputil=False,malloc=False,gc=False),
      logdir=str(tmp_path), envs=1,batch_size=1,batch_length=2,report_length=1,
      consec_report=1, action_mask_warmup=warmup, action_mask_actor_initial=False,
      seed_protocol={}, fg_full_checkpoint=True,fg_report_every_steps=5,
      fg_save_every_steps=10,fg_key_steps=[13],fg_reserve_gib=0,
      fg_checkpoint_identity=dict(test='fg_add_async_metrics'))
  def make_replay():
    return embodied.replay.Replay(2,64,chunksize=32,online=True,seed=7)
  def make_stream(replay,mode):
    return embodied.streams.Consec(embodied.streams.Stateless(partial(
        replay.sample,1,mode)),length=2 if mode=='train' else 1,
        consec=1,strict=mode=='train')
  train(lambda:agent,make_replay,lambda i:CountingEnv(),make_stream,
      lambda:elements.Logger(elements.Counter(),[elements.logger.TerminalOutput()]),args)
  assert agent.updates > 1 and agent.actor_enabled
  path = tmp_path/'ckpt/step_000000000013/action_mask_warmup.pkl'
  saved = pickle.loads(path.read_bytes())
  assert saved['actor_enabled'] and saved['last_add_support_sufficient']
