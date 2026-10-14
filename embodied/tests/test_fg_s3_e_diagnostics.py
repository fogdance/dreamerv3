"""Gradient observation must preserve every original optimizer update byte."""
import numpy as np

from test_fg_s3_e_model import raw_e, model, composition, train_batch
from dreamerv3.fg_s3_e_diagnostics import EAcceptanceModel
from dreamerv3.fg_s3_e_config import e_target_config
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from dreamerv3.fg_s3a_actor import parameter_digests
from embodied.run.fg_s3_source_transfer import _make_agent
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config


def test_gradient_observer_preserves_original_model_optimizer(raw_e,tmp_path):
    a=model(tmp_path/'original')
    b=_make_agent(e_target_config(small_config()),tmp_path/'observed',
        agent_cls=EAcceptanceModel,device='cpu',bc_lambda=1.,
        extra_agent_config=agent_extensions(dict(mode='a2')),composition=composition('a2'))
    assert parameter_digests(a.params)==parameter_digests(b.params)
    _,_,am=a.train(a.init_train(1),train_batch(a,raw_e,47))
    _,_,bm=b.train(b.init_train(1),train_batch(b,raw_e,47))
    assert parameter_digests(a.params)==parameter_digests(b.params)
    for k in am:np.testing.assert_array_equal(am[k],bm[k])
    assert all(float(bm['opt/e_'+name+'_grad_norm'])>0 for name in ('minute','context15','fusion'))
