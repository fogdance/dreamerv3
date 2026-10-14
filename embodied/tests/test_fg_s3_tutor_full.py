"""Full Tutor diagnostics must be observational, not another training update."""
import pickle
from pathlib import Path
import sys

import jax
import numpy as np

from embodied.run.fg_s3_source_transfer import make_agent
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config,gpu_batch
from gym_trading_env.research.fg_replay import signed,write_json

sys.path.insert(0,str(Path(__file__).parent))
from test_fg_s3_b2 import raw_batch, frozen


def test_same_actual_joint_update_with_and_without_gradient_sampling(raw_batch,tmp_path):
    def create(name,interval):
        return make_agent(small_config(),tmp_path/name,device='cpu',agent_cls=S3V2Agent,bc_lambda=1.,
            extra_agent_config=agent_extensions(dict(mode='tutor',tutor_hard_kappa=1.,
                                                      tutor_gradient_every_updates=interval)))
    a,b = create('sampled',2),create('plain',0)
    b.load_exact(pickle.loads(pickle.dumps(a.save_exact())))
    raw = {k:v for k,v in raw_batch.items() if k in a.spaces}
    ac,bc = a.init_train(1),b.init_train(1)
    frozen_before = parameter_digests(a.params)
    for i in range(2):
        ac,_,m = a.train(ac,gpu_batch(a,raw,900+i))
        bc,_,plain = b.train(bc,gpu_batch(b,raw,900+i))
        assert parameter_digests(a.params) == parameter_digests(b.params)
        assert m['tutor/gradient_sampled'] == (i == 1)
        assert m['opt/updates'] == plain['opt/updates'] == i+1
        assert m['opt/critic_grad_norm'] == m['v2/rl_active'] == 0
        if i == 1:
            for term in ('natural','balanced_reference','hard_balanced','pair'):
                assert m['tutor/grad/'+term+'_norm'] > 0
            np.testing.assert_allclose(m['tutor/grad/total_actor_norm'],m['opt/actor_grad_norm'],rtol=2e-4,atol=1e-5)
            assert m['tutor/difficulty_weight_min'] >= 1 and m['tutor/difficulty_weight_max'] <= 2
            assert m['tutor/difficulty_boosted_fraction'] > 0
    after = parameter_digests(a.params)
    assert all(frozen_before[k] == after[k] for k in after if frozen(k))
    assert a.save_exact()['counters'] == b.save_exact()['counters']
    saved = pickle.loads(pickle.dumps(a.save_exact()))
    a.load_exact(saved)
    ac,_,m = a.train(ac,gpu_batch(a,raw,902))
    assert not m['tutor/gradient_sampled'] and m['opt/updates'] == 3
    a.sync_policy_from_train()
    obs = {k:np.asarray(raw[k][:,0]) for k in a.obs_space}
    obs.update(is_first=np.array([True]),is_last=np.array([False]),action_mask=np.array([[1,0]],np.float32))
    from gym_trading_env.research.fg_single3_evaluation import ActorPolicy
    policy = ActorPolicy(a)
    policy(a.init_policy(1),obs)
    assert policy.last_diagnostics['tutor']['tutor/final_logit_add_minus_wait'] is None
    write_json(Path(__file__).resolve().parents[3]/'gym-trading-env/artifacts/fg_s3_v2/full_checks/tutor_cpu_diagnostics.json',
        signed(dict(status='PASS_CPU_SMALL_MODEL_TUTOR_DIAGNOSTICS',
            scope='controlled original-Env/native-Replay fixture; not production GPU training',
            diagnostic_changes_parameters=False,optimizer_updates_per_train_call=1,
            sampling_interval=2,restored_sampling_counter_continues=True,critic_frozen=True,
            illegal_final_margin_is_null=True,gpu='NOT_RUN')))
