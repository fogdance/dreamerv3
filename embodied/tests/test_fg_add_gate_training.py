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


def test_single3_online_transfer_syncs_policy_and_restarts_gate_observation(tmp_path):
  import json
  from embodied.run.train import train
  from embodied.run.action_mask_warmup import ActionMaskWarmup
  from embodied.tests.test_fg_full_checkpoint import checkpoint
  class TransferredAgent(CountingAgent):
    def __init__(self):
      super().__init__();self.used_gates=[];self.synced=False
    def sync_policy_from_train(self,reset_counter=False):
      assert reset_counter and self.updates==10000
      self.synced=True
    def train(self,carry,batch):
      self.used_gates.append(self.actor_enabled)
      carry,outs,metrics=super().train(carry,batch)
      for prefix in ('post','prior'):
        metrics[f'avail/{prefix}_add_positives']=32
        metrics[f'avail/{prefix}_add_true_positives']=32
      return carry,outs,metrics
  gate=elements.Config(dict(enabled=True,min_train_samples=1,consecutive_reports=5,
      regression_reports=5,stop_on_regression=True,post_accuracy=.995,post_exact=.99,
      prior_accuracy=.99,prior_exact=.98,prior_fpr=.005,img_fallback=.0001,
      add_recall_enabled=True,min_add_positives=2,post_add_recall=.8,prior_add_recall=.8))
  # Passing offline observations must not count towards the online window.
  warmup=ActionMaskWarmup.for_single3_1m(gate)
  warmup.reports=20;warmup.consecutive_passes=20
  cp,step=checkpoint(tmp_path/'source')
  cp.agent=CountingAgent();cp.agent.updates=10000;cp.action_mask_warmup=warmup
  step.load(10000);cp.save(final=True)
  source=cp.latest()
  before=(source/'agent.pkl').read_bytes()
  receipt=tmp_path/'transfer.json'
  receipt.write_text(json.dumps(dict(checkpoint=str(source),source_kind='single3_offline')))
  agent=TransferredAgent()
  args=elements.Config(steps=120,train_ratio=4,log_every=1e9,report_every=1e9,save_every=1e9,
      report_batches=1,from_checkpoint='',fg_stage_transfer=str(receipt),env_checkpoint=True,debug=True,
      usage=dict(psutil=False,nvsmi=False,gputil=False,malloc=False,gc=False),logdir=str(tmp_path/'online'),
      envs=1,batch_size=1,batch_length=2,report_length=1,consec_report=1,
      action_mask_warmup=gate,action_mask_actor_initial=False,seed_protocol={},fg_full_checkpoint=True,
      fg_report_every_steps=5,fg_save_every_steps=50,fg_key_steps=[0,120],fg_reserve_gib=0,
      fg_checkpoint_identity=dict(test='single3_online_phase'))
  make_replay=lambda:embodied.replay.Replay(2,64,chunksize=32,online=True,seed=7)
  def stream(replay,mode):
    return embodied.streams.Consec(embodied.streams.Stateless(partial(replay.sample,1,mode)),
        length=2 if mode=='train' else 1,consec=1,strict=mode=='train')
  train(lambda:agent,make_replay,lambda i:CountingEnv(),stream,
      lambda:elements.Logger(elements.Counter(),[lambda _:None]),args)
  assert agent.synced and agent.updates>10000
  assert not agent.used_gates[0] and any(agent.used_gates)
  final=tmp_path/'online/ckpt/step_000000000120'
  state=pickle.loads((final/'action_mask_warmup.pkl').read_bytes())
  assert state['actor_enabled'] and state['switched_step']>=100
  initial=tmp_path/'online/ckpt/step_000000000000'
  assert pickle.loads((initial/'agent.pkl').read_bytes())['updates']==10000
  assert pickle.loads((initial/'action_mask_warmup.pkl').read_bytes())['consecutive_passes']==0
  assert (source/'agent.pkl').read_bytes()==before
