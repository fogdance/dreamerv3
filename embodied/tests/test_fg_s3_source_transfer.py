"""Actual original CPU Agent and atomic S3-A format; not production sources."""
from pathlib import Path
import copy
import pickle
import sys

import jax
import numpy as np
import pytest

GYM=Path(__file__).resolve().parents[3]/'gym-trading-env'
sys.path.insert(0,str(GYM/'tests/entry_v1'))
from test_fg_s3a_labels import native_dataset
from gym_trading_env.research.fg_s3a_labels import build_labels,load_labels
from gym_trading_env.research.fg_s3_teacher_catalog import build_catalog,load_catalog
from gym_trading_env.research.fg_replay import iter_replay_rows
from embodied.run.fg_s3a_checkpoint import diagnostic_source,save_bc,audit_bc
from embodied.run.fg_s3a_features import teacher_step
from embodied.run.fg_s3a_train import Sampler
from embodied.run.fg_s3_source_admission import inspect_source,prepare_transfer,transfer_control_state
from embodied.run.fg_s3_source_transfer import make_agent,apply_and_audit_transfer
from dreamerv3.fg_s3a_actor import ActorBC,parameter_digests


@pytest.fixture(scope='module')
def source_bundle(tmp_path_factory,request):
    root=tmp_path_factory.mktemp('actual_source')
    dataset,oracle,trace,tm=native_dataset(root/'native')
    labels=root/'labels';build_labels(dataset,oracle,labels)
    build_catalog(dataset,oracle,labels,root/'catalog',diagnostic=True)
    c=load_catalog(root/'catalog')
    a,source=diagnostic_source(load_labels(labels),root/'base')
    if getattr(request,'param',False):
        # Explicit diagnostic READY source fixture; no production Gate bypass.
        a.set_avail_actor_enabled(True)
    # Nonzero moments/norms exercise preservation beyond a fresh empty optimizer.
    for k,v in list(a.params.items()):
        if k.startswith(('opt/state/','retnorm/')):
            a.params[k]=jax.device_put(np.full(v.shape,3 if np.issubdtype(v.dtype,np.integer) else .125,dtype=v.dtype),v.sharding)
    a.sync_policy_from_train()
    node=Path(source['path']);(node/'agent.pkl').write_bytes(pickle.dumps(a.save_exact(),protocol=5))
    import json
    from embodied.run.fg_checkpoint import file_hash
    meta=json.loads((node/'manifest.json').read_text());meta['files']['agent.pkl']=file_hash(node/'agent.pkl')
    (node/'manifest.json').write_text(json.dumps(meta))
    source['manifest_sha256']=file_hash(node/'manifest.json')
    bc=ActorBC(a);step=next(iter_replay_rows(trace,tm))[1]
    _,x,_=teacher_step(a,a.init_policy(1),step)
    for _ in range(2):bc.update(np.repeat(x[None],2,axis=0),np.ones((2,2),np.float32),np.ones(2,np.int32))
    from embodied.run.fg_s3a_checkpoint import code_identity,package_versions
    identity=dict(source_manifest_sha256=source['manifest_sha256'],source_identity_hash=source['identity_hash'],
                  label_manifest_sha256=c.manifest['labels_manifest_sha256'],diagnostic=True,
                  code_sha256=code_identity(),packages=package_versions())
    bnode=save_bc(root/'bc',bc,Sampler(2,0),identity)
    return dict(root=root,catalog=c,base=node,bc=bnode,config=source['config'],
                labels=labels,trace=trace,tm=tm)


def test_actual_three_modes_inherit_complete_state_and_reject_foreign(source_bundle):
    b=source_bundle;c=b['catalog']
    target=make_agent(b['config'],b['root']/'target')
    initial=inspect_source('direct_init',catalog=c,seed=0,config=b['config'])
    assert apply_and_audit_transfer(target,initial)['kind']=='direct_init'
    direct=inspect_source('direct_full',catalog=c,path=b['base'])
    assert transfer_control_state(direct)['complete']
    ds=prepare_transfer(direct);rd=apply_and_audit_transfer(target,direct)
    assert rd['params_after']==parameter_digests(ds['params'])
    after=inspect_source('after_bc_s3a',catalog=c,path=b['bc'],base_checkpoint=b['base'])
    saved=prepare_transfer(after);ra=apply_and_audit_transfer(target,after)
    assert ra['baseline_non_actor_optimizer_equality']=='PASS'
    assert ra['params_after']==parameter_digests(saved['params'])
    assert ra['joint_opt']==rd['joint_opt'] and ra['gate_norm']==rd['gate_norm']
    assert not ra['independent_bc_optimizer_imported']
    assert any(ra['params_after'][k]!=rd['params_after'][k] for k in ra['params_after'] if k.startswith('pol/'))
    with pytest.raises(ValueError,match='Resume'):inspect_source('resume_s3c',catalog=c)
    with pytest.raises(ValueError,match='NOT_SUPPORTED'):inspect_source('after_bc_s3b',catalog=c)
    with pytest.raises(ValueError,match='schema|shapes|keys'):prepare_transfer(after,{'foreign':((1,),'float32')})
    corrupt=copy.deepcopy(after);corrupt['origin']['step']+=1
    with pytest.raises(ValueError,match='corrupt'):prepare_transfer(corrupt)
    wrong=copy.copy(c);wrong.manifest=copy.deepcopy(c.manifest);wrong.manifest['source']['dataset_sha256']='foreign'
    with pytest.raises(ValueError,match='identity'):inspect_source('direct_full',catalog=wrong,path=b['base'])
    # Isolated flag-contract test. It does not authenticate a production model.
    real_flag=copy.copy(c);real_flag.manifest=copy.deepcopy(c.manifest)
    real_flag.manifest['diagnostic']=False
    spec=inspect_source('direct_init',catalog=real_flag,seed=0,config=b['config'],diagnostic=True)
    assert spec['origin']['diagnostic'] and spec['catalog_manifest_sha256']==c.manifest_sha256
    with pytest.raises(ValueError,match='boolean'):
        inspect_source('direct_init',catalog=c,seed=0,config=b['config'],diagnostic='false')


def test_no_inference_source_and_frozen_snapshot(source_bundle,tmp_path,monkeypatch):
    from embodied.run.fg_s3_source_admission import freeze_source
    from embodied.run import fg_s3_source_transfer as transfer
    b=source_bundle
    frozen=freeze_source(b['base'],tmp_path/'snapshot')
    spec=inspect_source('direct_full',catalog=b['catalog'],path=frozen['snapshot'])
    state=prepare_transfer(spec)
    # Actual neural component test for the save_exact inference-disabled state.
    state=dict(state,policy_params=None,pending_sync=None)
    monkeypatch.setattr(transfer,'prepare_transfer',lambda s,schema:state)
    agent=make_agent(b['config'],tmp_path/'target')
    receipt=apply_and_audit_transfer(agent,spec)
    assert receipt['policy_initialized_from_train']
    assert parameter_digests(agent.params)==parameter_digests(state['params'])
    assert agent.save_exact()['counters']==state['counters']


def test_missing_baseline_requires_authenticated_features(source_bundle,tmp_path):
    import shutil
    from gym_trading_env.research.fg_replay import read_manifest,write_json,signed,digest
    b=source_bundle
    base=inspect_source('direct_full',catalog=b['catalog'],path=b['base'])['origin']
    features=tmp_path/'features';features.mkdir()
    write_json(features/'manifest.json',signed(dict(source=dict(base,path=str(tmp_path/'unavailable')))))
    node=tmp_path/'bc';shutil.copytree(b['bc'],node)
    m=read_manifest(node/'manifest.json');m.pop('content_hash')
    m['identity']['feature_manifest_sha256']=digest(features/'manifest.json')
    write_json(node/'manifest.json',signed(m))
    with pytest.raises(ValueError,match='provenance'):
        inspect_source('after_bc_s3a',catalog=b['catalog'],path=node,config=b['config'])
    spec=inspect_source('after_bc_s3a',catalog=b['catalog'],path=node,features=features)
    assert spec['origin']['baseline_optimizer_equality']=='NOT_VERIFIED'
    target=make_agent(b['config'],tmp_path/'target')
    receipt=apply_and_audit_transfer(target,spec)
    assert receipt['baseline_non_actor_optimizer_equality']=='NOT_VERIFIED'
    assert receipt['params_before']==receipt['params_after']
