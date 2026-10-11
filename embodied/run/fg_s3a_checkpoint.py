"""Read-only source admission and atomic S3-A checkpoint packages."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import json
import os
import pickle
import uuid

import elements
import jax
import numpy as np

from embodied.run.fg_checkpoint import file_hash, plain, validate_bundle, learning_spec
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import check_files, read_manifest, signed, write_json

PROTOCOL='FG_S3A_BC_V1'
DIAGNOSTIC='FG_S3A_DIAGNOSTIC_SOURCE_V1'


def code_identity():
    root=Path(__file__).resolve().parents[2]
    names=['dreamerv3/fg_s3a_actor.py','dreamerv3/fg_s3a_agent.py','embodied/run/fg_s3a_checkpoint.py',
           'embodied/run/fg_s3a_features.py','embodied/run/fg_s3a_train.py',
           'embodied/run/fg_s3a_eval.py','scripts/fg_s3a.py']
    result={n:file_hash(root/n) for n in names}
    # Bind the actual Gym modules used for native decoding, identity and labels.
    from gym_trading_env.research import fg_replay,fg_s3a_labels
    from gym_trading_env.envs import entry_contract
    from gym_trading_env.rewards import single3_reward
    for module in (fg_replay,fg_s3a_labels,entry_contract,single3_reward):
        result[module.__name__.replace('.','/')+'.py']=file_hash(module.__file__)
    return result


def package_versions():
    return {n:importlib.metadata.version(n) for n in
            ('jax','jaxlib','ninjax','optax','numpy','elements','gymnasium','pandas')}


def validate_source(path, labels, *, diagnostic=False):
    path=Path(path).resolve()
    metadata=json.loads((path/'manifest.json').read_text())
    tag=read_manifest(path.parent.parent/'s3a_diagnostic_identity.json') if diagnostic else json.loads(
        (path.parent.parent/'single3_identity.json').read_text())
    identity=tag['identity'];expected_hash=fingerprint(identity)
    if expected_hash!=tag['identity_hash']:
        raise ValueError('Source model identity hash mismatch')
    protocol=DIAGNOSTIC if diagnostic else 'FG_SINGLE3_OFFLINE_V1'
    if (metadata['identity']!=dict(protocol=protocol,identity_hash=expected_hash)
            or identity.get('candidate')!='C1' or identity.get('pool')!='C1_natural_oracle'
            or identity['dataset_sha256']!=labels['source']['dataset_sha256']
            or identity['reward_params_hash']!=labels['source']['reward_params_hash']
            or identity['observation_identity']!=labels['traces'][0]['observation_identity']
            or identity['model_seed']!=0 or identity['replay_seed']!=202):
        raise ValueError('Source C1 model/dataset/Reward/Obs/seed identity mismatch')
    # Fail early without loading an in-flight production agent payload.
    if not diagnostic:
        if identity.get('profile')!='C1_M1_1M' or metadata['step']!=62500 or not metadata['protected']:
            raise ValueError('Completed protected C1_M1_1M checkpoint (62500 updates) required')
        summary=json.loads((path.parent.parent/'training_summary.json').read_text())
        if (summary.get('status')!='TRAINING_COMPLETE' or summary['last_update']!=62500
                or summary['identity_hash']!=expected_hash
                or summary['checkpoint_manifest_sha256']!=file_hash(path/'manifest.json')
                or Path(summary['checkpoint']).resolve()!=path):
            raise ValueError('Source completion summary is not bound to final checkpoint')
    meta=validate_bundle(path,metadata['identity'])
    from dreamerv3.configuration import read_config_yaml
    config=read_config_yaml(path/'config.yaml')
    if (config['agent']!=identity['agent_config'] or config['agent']['horizon']!=333
            or not config['agent']['contdisc'] or plain(learning_spec(config))!=identity['learning']):
        raise ValueError('Source architecture/horizon config differs from identity')
    root=Path(__file__).resolve().parents[2]
    for rel,checksum in identity['code_sha256'].items():
        if file_hash(root/rel)!=checksum:
            raise ValueError('Source core implementation changed: '+rel)
    for name,version in identity['packages'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('Source dependency version mismatch: '+name)
    return dict(path=str(path),manifest_sha256=file_hash(path/'manifest.json'),
                identity_hash=expected_hash,identity=identity,config=config,
                diagnostic=diagnostic,step=meta['step'])


def make_agent(config, output, *, device='cpu'):
    from dreamerv3.fg_s3a_agent import S3AAgent
    from gym_trading_env.research.fg_replay import FIELDS
    obs={k:elements.Space(dtype,shape) for k,(shape,dtype) in FIELDS.items() if k!='action'}
    act={'action':elements.Space(np.int32,(),0,2)}
    cfg=elements.Config(config)
    jcfg=dict(cfg.jax)
    jcfg.update(platform='cpu' if device=='cpu' else 'cuda',precompile=False,profiler=False,
                prealloc=False,enable_policy=True,transfer_guard=False)
    return S3AAgent(obs,act,elements.Config(**dict(cfg.agent),logdir=str(output),seed=cfg.seed,
        jax=jcfg,batch_size=cfg.batch_size,batch_length=cfg.batch_length,
        replay_context=cfg.replay_context,report_length=cfg.report_length,
        replica=0,replicas=1))


def load_source(source, output, *, device='cpu'):
    agent=make_agent(source['config'],output,device=device)
    state=pickle.loads((Path(source['path'])/'agent.pkl').read_bytes())
    if state['counters']['updates']!=source['step']:
        raise ValueError('Source agent update counter differs from checkpoint')
    # No regex or partial loads: exact keys and dtype/shape are required.
    if set(state['params'])!=set(agent.params) or any(
            np.shape(v)!=agent.params[k].shape or str(v.dtype)!=str(agent.params[k].dtype)
            for k,v in state['params'].items()):
        raise ValueError('Source parameter schema differs from original Agent')
    agent.load(state,restore_batches=True)
    agent.sync_policy_from_train(reset_counter=True)
    from dreamerv3.fg_s3a_actor import parameter_digests
    if parameter_digests(agent.params)!=parameter_digests(state['params']):
        raise RuntimeError('Source was not loaded byte-exactly')
    return agent


def save_bc(directory, bc, sampler, identity):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    dest=root/f'bc_{bc.bc_updates:012d}'
    if dest.exists():
        raise FileExistsError('BC checkpoint already exists')
    pending=root/('.pending_'+uuid.uuid4().hex);pending.mkdir()
    audit=bc.freeze_audit(require_actor_change=bc.bc_updates>0)
    # Preserve Optax namedtuple nodes. The legacy host() recursively turns them
    # into ordinary tuples, which destroys the optimizer's PyTree contract.
    opt_state=jax.tree.map(lambda a:np.array(jax.device_get(a),copy=True),bc.s3a_bc_opt)
    state=dict(agent=bc.agent.save_exact(),s3a_bc_opt=opt_state,
               bc_updates=bc.bc_updates,sampler=sampler.save(),source_digests=bc.source_digests)
    with (pending/'state.pkl').open('wb') as stream:
        pickle.dump(state,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    write_json(pending/'freeze_audit.json',audit)
    write_json(pending/'manifest.json',signed(dict(protocol=PROTOCOL,identity=identity,
        bc_updates=bc.bc_updates,files={p.name:file_hash(p) for p in pending.iterdir() if p.is_file()})))
    (pending/'done').write_text('')
    pending.rename(dest)
    pointer=root/'.latest.pending';pointer.write_text(dest.name+'\n');os.replace(pointer,root/'latest')
    return dest


def validate_bc(path, identity=None):
    path=Path(path);m=read_manifest(path/'manifest.json')
    if not (path/'done').is_file() or m['protocol']!=PROTOCOL or (identity is not None and m['identity']!=identity):
        raise ValueError('Incomplete or foreign S3-A checkpoint identity')
    if {p.name for p in path.iterdir()}!=set(m['files'])|{'manifest.json','done'}:
        raise ValueError('BC checkpoint file set mismatch')
    check_files({str(path/p):h for p,h in m['files'].items()})
    return m


def audit_bc(path, identity=None):
    """Recompute the saved full parameter proof without executing the model."""
    m=validate_bc(path,identity)
    if (m['identity']['code_sha256']!=code_identity()
            or m['identity']['packages']!=package_versions()):
        raise ValueError('BC implementation identity changed')
    state=pickle.loads((Path(path)/'state.pkl').read_bytes())
    proof=json.loads((Path(path)/'freeze_audit.json').read_text())
    from dreamerv3.fg_s3a_actor import parameter_digests
    current=parameter_digests(state['agent']['params']);before=state['source_digests']
    changed=sorted(k for k in before if before[k]!=current.get(k))
    if (set(current)!=set(before) or any(not k.startswith('pol/') for k in changed)
            or (m['bc_updates']>0 and not changed) or state['bc_updates']!=m['bc_updates']
            or proof['source_digests']!=before or proof['current_digests']!=current
            or proof['changed_keys']!=changed or proof['bc_updates']!=state['bc_updates']
            or proof['original_updates']!=state['agent']['counters']['updates']
            or proof['original_batches']!=state['agent']['counters']['batches']):
        raise ValueError('BC saved parameter freeze proof differs')
    policy=state['agent']['policy_params']
    policy_digests=parameter_digests(policy) if policy is not None else None
    if (state['agent']['pending_sync'] is not None or policy is None
            or any(policy_digests[k]!=current[k] for k in policy)):
        raise ValueError('BC saved inference parameters differ from updated model')
    return dict(m,freeze_audit=proof)


def restore_bc(path,bc,sampler,identity):
    m=validate_bc(path,identity)
    state=pickle.loads((Path(path)/'state.pkl').read_bytes())
    if state['bc_updates']!=m['bc_updates'] or state['source_digests']!=bc.source_digests:
        raise ValueError('BC source fingerprint or update count differs')
    if jax.tree.structure(state['s3a_bc_opt'])!=jax.tree.structure(bc.s3a_bc_opt):
        raise ValueError('BC optimizer structure differs')
    for saved,expected in zip(jax.tree.leaves(state['s3a_bc_opt']),jax.tree.leaves(bc.s3a_bc_opt)):
        if saved.shape!=expected.shape or saved.dtype!=expected.dtype or not np.isfinite(saved).all():
            raise ValueError('BC optimizer shape/dtype/finite contract differs')
    params=state['agent']['params']
    if (set(params)!=set(bc.source_shapes) or any((v.shape,str(v.dtype))!=bc.source_shapes[k]
            for k,v in params.items())):
        raise ValueError('BC parameter schema differs from source')
    from dreamerv3.fg_s3a_actor import parameter_digests
    restored=parameter_digests(params)
    if any(restored[k]!=bc.source_digests[k] for k in restored if k not in bc.actor_keys):
        raise ValueError('BC checkpoint contains changed frozen state')
    if any(state['agent']['counters'][k]!=v for k,v in bc.source_counters.items()):
        raise ValueError('BC checkpoint contains changed joint counters')
    bc.agent.load_exact(state['agent'])
    bc.agent.sync_policy_from_train()
    bc.s3a_bc_opt=jax.tree.map(jax.device_put,state['s3a_bc_opt'])
    bc.bc_updates=int(state['bc_updates']);sampler.load(state['sampler'])
    bc.frozen_refs={k:v for k,v in bc.agent.params.items() if k not in bc.actor_keys}
    bc.freeze_audit(require_actor_change=bc.bc_updates>0)
    return m


def diagnostic_source(labels,path,*,compute_dtype='float32'):
    """Real original Agent with small architecture; never a production result."""
    from dreamerv3.configuration import read_config_yaml
    import yaml
    if compute_dtype not in ('float32','bfloat16'):raise ValueError('Unsupported diagnostic dtype')
    root=Path(__file__).resolve().parents[2]
    defaults=read_config_yaml(root/'dreamerv3/configs.yaml')['defaults']
    defaults['jax']['transfer_guard']=False
    cfg=elements.Config(defaults).update({
        'jax.platform':'cpu','jax.compute_dtype':compute_dtype,'jax.precompile':False,
        'jax.profiler':False,'jax.prealloc':False,'jax.transfer_guard':False,
        'batch_size':2,'batch_length':4,'report_length':4,'replay_context':1,
        'agent.replay_context_mode':'recompute_prefix','agent.update_diagnostics':True,
        'agent.enc.simple.layers':1,'agent.enc.simple.units':16,
        'agent.dec.simple.layers':1,'agent.dec.simple.units':16,
        'agent.dyn.rssm.deter':32,'agent.dyn.rssm.hidden':16,
        'agent.dyn.rssm.stoch':2,'agent.dyn.rssm.classes':4,'agent.dyn.rssm.blocks':4,
        'agent.policy.layers':1,'agent.policy.units':16,'agent.value.layers':1,'agent.value.units':16,
        'agent.rewhead.units':16,'agent.conhead.units':16,'agent.availhead.units':16,
        'agent.imag_length':3,'agent.horizon':333,'agent.contdisc':True,
        'agent.avail_actor_enabled':False,'agent.avail_always_available_index':0})
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    agent=make_agent(plain(cfg),path)
    identity=dict(candidate='C1',pool='C1_natural_oracle',model_seed=0,replay_seed=202,
        dataset_sha256=labels['source']['dataset_sha256'],reward_params_hash=labels['source']['reward_params_hash'],
        observation_identity=labels['traces'][0]['observation_identity'],agent_config=plain(cfg.agent),
        code_sha256={n:file_hash(root/n) for n in ['dreamerv3/agent.py','dreamerv3/rssm.py',
            'embodied/jax/agent.py','embodied/jax/opt.py','embodied/jax/outs.py','dreamerv3/configs.yaml']},
        packages={n:importlib.metadata.version(n) for n in ['jax','jaxlib','ninjax','optax','numpy','elements']},
        learning=plain(learning_spec(cfg)),source_kind='RANDOM_SMALL_ORIGINAL_AGENT_CPU_DIAGNOSTIC')
    ih=fingerprint(identity)
    write_json(path/'s3a_diagnostic_identity.json',signed(dict(identity=identity,identity_hash=ih)))
    node=path/'ckpt/step_000000000000';node.mkdir(parents=True)
    (node/'agent.pkl').write_bytes(pickle.dumps(agent.save_exact(),protocol=5))
    (node/'step.pkl').write_bytes(pickle.dumps(0,protocol=5))
    (node/'config.yaml').write_text(yaml.safe_dump(plain(cfg)))
    (node/'manifest.json').write_text(json.dumps(dict(schema=1,protocol='FG_FULL_V1',step=0,
        protected=True,identity=dict(protocol=DIAGNOSTIC,identity_hash=ih),
        files={p.name:file_hash(p) for p in node.iterdir()}),indent=2))
    (node/'done').write_text('')
    return agent,validate_source(node,labels,diagnostic=True)
