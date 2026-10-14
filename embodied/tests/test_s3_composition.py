"""Actual small CPU updates/model restore; controlled market, no E model."""
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

from embodied.run.fg_checkpoint import host
from embodied.run.fg_s3_source_transfer import make_agent
from dreamerv3.fg_s3_composition import KEY, resolve_composition
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config, gpu_batch

sys.path.insert(0,str(Path(__file__).parent))
from test_fg_s3_b2 import raw_batch, frozen


@pytest.mark.parametrize('mode', ['a2','tutor','b2'])
def test_new_composition_and_legacy_have_identical_actual_updates_and_restore(raw_batch,tmp_path,mode):
    resolved=resolve_composition({'mode':mode})
    settings=dict(mode=mode,tutor_hard_kappa=1.,b2_mix_warmup=100,b2_mix_ramp=100,b2_min_event_updates=100)
    extension=agent_extensions(settings)
    def model(name, composition=None):
        return make_agent(small_config(),tmp_path/name,device='cpu',agent_cls=S3V2Agent,
            bc_lambda=1.,extra_agent_config=extension,composition=composition)
    old,new=model('legacy'),model('composition',resolved)
    new.load_exact(pickle.loads(pickle.dumps(old.save_exact())))
    assert parameter_digests(old.params)==parameter_digests(new.params)
    before=parameter_digests(new.params)
    raw={k:v for k,v in raw_batch.items() if k in old.spaces}
    oc,nc=old.init_train(1),new.init_train(1)
    # The original 1000-update optimizer ramp starts at learning rate zero.
    for seed in (1300,1301):
        oc,_,om=old.train(oc,gpu_batch(old,raw,seed))
        nc,_,nm=new.train(nc,gpu_batch(new,raw,seed))
        assert parameter_digests(old.params)==parameter_digests(new.params)
        assert om.keys()==nm.keys()
        for k in om:np.testing.assert_allclose(om[k],nm[k],atol=0,rtol=0)
    after=parameter_digests(new.params)
    assert any(before[k]!=after[k] for k in before if k.startswith('pol/'))
    assert any(before[k]!=after[k] for k in before if k.startswith(('enc/','dyn/')))
    assert all(before[k]==after[k] for k in before if frozen(k))
    new.sync_policy_from_train()
    saved=tmp_path/'controlled_model.pkl'
    saved.write_bytes(pickle.dumps(dict(agent=new.save_exact(),carry=host(nc))))
    restored=model('restored',resolved)
    state=pickle.loads(saved.read_bytes());restored.load_exact(state['agent'])
    assert parameter_digests(new.params)==parameter_digests(restored.params)
    nc,_,nm=new.train(nc,gpu_batch(new,raw,1302))
    _,_,rm=restored.train(restored.restore_carry(state['carry']),gpu_batch(restored,raw,1302))
    assert parameter_digests(new.params)==parameter_digests(restored.params)
    assert nm['opt/updates']==rm['opt/updates']==3
