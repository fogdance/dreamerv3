"""Own E FullCheckpoint exact CPU resume with original Driver/Replay/Env."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import elements
import embodied
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_fg_s3_e_model import model, composition, train_batch
from e_model_fixture import collector_fixture
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config
from dreamerv3.fg_s3_e_config import e_target_config
from dreamerv3.fg_s3a_actor import parameter_digests
from embodied.run.fg_checkpoint import FullCheckpoint, RuntimeState, StepSchedule, validate_bundle, plain, file_hash
from embodied.run.fg_s3_v2_train import Counters
from embodied.run.fg_s3_e_source import make_binding
from embodied.run.fg_s3_gate import gate_policy, make_gate_controller


@pytest.mark.parametrize('mode',['a2','tutor','b2'])
def test_EM15_full_checkpoint_continuous_vs_interrupted_exact(mode,tmp_path):
    a,b=model(tmp_path/'a',mode),model(tmp_path/'b',mode)
    ca,provider=collector_fixture(tmp_path/'ca',agent=a)
    cb=None
    try:
        ca.advance(70)
        cfg=e_target_config(small_config())
        binding=make_binding(cfg,composition(mode),provider.manifest)
        identity=dict(protocol='FG_S3_E_CPU_FULL_CHECKPOINT_CONTRACT_V1',
            scope='CPU_SYNTHETIC_CONTRACT_ONLY',resolved_composition=composition(mode),
            e_binding=binding,teacher_identity=ca.oracle.identity_hash,
            outcome_identity=ca.outcome_catalog.identity_hash,
            agent_extension=plain({k:a.config[k] for k in a.config if k.startswith(('tutor_','b2_'))}),
            clock_protocol='S3_MARKET_MINUTE')
        def wire(agent,collector,name):
            stream=embodied.streams.Consec(embodied.streams.Stateless(lambda:collector.sample(1)),
                length=64,consec=1,prefix=1,strict=True)
            box=[agent.init_train(1)]
            runtime=RuntimeState(StepSchedule(100,100),SimpleNamespace(),[stream],
                lambda:box[0],lambda c:box.__setitem__(0,agent.restore_carry(c)),{},[0])
            counters=Counters(mode)
            gate_policy_value=gate_policy(cfg,diagnostic=True)
            gate=make_gate_controller(cfg,gate_policy_value,dict(gate=None,aggregate=None,samples=0),False)
            cp=FullCheckpoint(tmp_path/name,elements.Counter(70),identity,[stream],reserve_gib=0,keep_all=True)
            cp.agent,cp.replay,cp.driver,cp.collector=agent,collector.replay,collector.driver,collector
            cp.step,cp.runtime,cp.v2_counters,cp.action_mask_warmup=cp._step,runtime,counters,gate
            return cp,stream,box,counters,gate
        cp,stream,box,counters,gate=wire(a,ca,'checkpoint')
        for seed in (200,201):
            box[0],_,_=a.train(box[0],train_batch(a,next(iter(stream)),seed))
            counters.state['all_updates']+=1
        a.sync_policy_from_train()
        cp.save();node=cp.latest();manifest=validate_bundle(node,identity)
        assert manifest['identity']['resolved_composition']['support_state']=='ready'
        assert all(any(k.startswith(p) for k in a.params) for p in
                   ('enc/minute/','enc/context15/','enc/fusion/'))
        for wrong in ({**identity,'resolved_composition':composition('b2' if mode!='b2' else 'a2')},
                      {**identity,'e_binding':{**binding,'data_manifest_sha256':'foreign'}}):
            with pytest.raises(ValueError,match='identity mismatch'):validate_bundle(node,wrong)
        other=FixedDirectionEntryEnv(df=ca.env._raw.copy(),config_path=str(tmp_path/'ca/market/env.yaml'))
        cb,_=collector_fixture(tmp_path/'cb',agent=b,env=other,provider=provider,
            query=ca.oracle,outcomes=ca.outcome_catalog)
        cb.journal.write_bytes(ca.journal.read_bytes())
        restored,bs,bb,bcounters,bgate=wire(b,cb,'restored')
        restored.load(node)
        assert parameter_digests(a.params)==parameter_digests(b.params)
        assert parameter_digests(a.save_exact()['policy_params'])==parameter_digests(b.save_exact()['policy_params'])
        assert counters.save()==bcounters.save() and gate.save()==bgate.save()
        assert ca.records==cb.records and ca.pending==cb.pending
        ca.advance(75);cb.advance(75)
        assert ca.records==cb.records and ca.pending==cb.pending
        next_metrics=[]
        for seed in (202,203,204):
            x,y=next(iter(stream)),next(iter(bs))
            for k in x:np.testing.assert_array_equal(x[k],y[k])
            box[0],_,am=a.train(box[0],train_batch(a,x,seed))
            bb[0],_,bm=b.train(bb[0],train_batch(b,y,seed))
            counters.state['all_updates']+=1;bcounters.state['all_updates']+=1
            assert parameter_digests(a.params)==parameter_digests(b.params)
            for k in am:np.testing.assert_array_equal(am[k],bm[k])
            next_metrics.append({k:float(np.asarray(v)) for k,v in am.items() if np.asarray(v).size==1})
        assert a.save_exact()['counters']==b.save_exact()['counters']
        assert counters.save()==bcounters.save() and int(a.n_updates)==5
        a.sync_policy_from_train();b.sync_policy_from_train()
        assert parameter_digests(a.save_exact()['policy_params'])==parameter_digests(b.save_exact()['policy_params'])
        receipt=dict(mode=mode,scope='CPU_SYNTHETIC_CONTRACT_ONLY',checkpoint=str(node),
            manifest_sha256=file_hash(node/'manifest.json'),identity=identity,
            model_updates_before=2,model_updates_after=5,env_steps_before=70,env_steps_after=75,
            parameters=parameter_digests(a.params),agent_counters=a.save_exact()['counters'],
            next_metrics=next_metrics,exact_model_optimizer=True,exact_policy=True,
            exact_replay_sample=True,exact_driver_env_sidecar=True,exact_rng_and_counters=True,
            production_resume='SYNTHETIC_CHECKPOINT_NOT_FOR_FORMAL_TRAINING')
        (tmp_path/'checkpoint_evidence.json').write_text(json.dumps(receipt,indent=2))
    finally:
        ca.close()
        if cb:cb.close()
