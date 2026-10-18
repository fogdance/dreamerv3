"""Explicit F1→F2 parameter handoff, distinct from FullCheckpoint resume."""
from pathlib import Path
import pickle
import numpy as np
from .fg_checkpoint import validate_bundle,plain
from gym_trading_env.research.fg_replay import read_manifest,digest,signed
from gym_trading_env.envs.entry_contract import fingerprint

def training_evidence(root,identity,agent):
    """FORMAL requires committed update records and actual WM parameter change."""
    import json
    root=Path(root);budget=identity['preparation']['optimizer_updates']
    log=root/'updates.jsonl'
    rows=[json.loads(line) for line in log.read_text().splitlines()]
    if len(rows)!=budget:raise ValueError('F2_F1_TRAINING_RECORDS_INCOMPLETE')
    for i,row in enumerate(rows,1):
        m=row['metrics']
        if (row['local_update']!=i or row['agent_update']!=i or m['f1/optimizer_updates']!=i
                or m['f1/finite'] is not True or m['f1/target_positions']<=0
                or m['opt/updates']!=i or not np.isfinite(m['opt/loss']) or not np.isfinite(m['opt/grad_norm'])
                or m['f1/actor_updates']!=0 or m['f1/teacher_updates']!=0):
            raise ValueError('F2_F1_TRAINING_RECORDS_INVALID')
    initial=root/'ckpt/step_000000000000';validate_bundle(initial,identity)
    state=pickle.loads((initial/'agent.pkl').read_bytes())
    if state['counters']['updates']!=0:raise ValueError('F2_F1_INITIAL_STATE_INVALID')
    if (state['params'].keys()!=agent['params'].keys() or any(
            state['params'][k].shape!=v.shape or state['params'][k].dtype!=v.dtype
            or not np.isfinite(v).all() for k,v in agent['params'].items())):
        raise ValueError('F2_F1_FULL_STATE_SCHEMA_OR_FINITE_INVALID')
    from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
    before={k:v for k,v in state['params'].items() if k.startswith(WM_GROUPS)}
    after={k:v for k,v in agent['params'].items() if k.startswith(WM_GROUPS)}
    if before.keys()!=after.keys() or any(before[k].shape!=after[k].shape or before[k].dtype!=after[k].dtype for k in before):raise ValueError('F2_F1_WM_SCHEMA_CHANGED')
    changed={p:[k for k in before if k.startswith(p) and before[k].tobytes()!=after[k].tobytes()] for p in WM_GROUPS}
    if not any(changed.values()):raise ValueError('F2_F1_FOUNDATION_PARAMETERS_NOT_TRAINED')
    return dict(update_log_sha256=digest(log),initial_manifest_sha256=digest(initial/'manifest.json'),
        verified_updates=budget,changed_wm_keys=changed)

def inspect_f1(settings,*,allow_engineering=False):
    root=Path(settings['source_f1_run']);node=Path(settings['source_f1_checkpoint'])
    if not node.is_relative_to(root/'ckpt'):raise ValueError('F2_F1_NODE_OUTSIDE_RUN')
    identity=read_manifest(root/'run_identity.json')['identity']
    meta=validate_bundle(node,identity)
    if (identity.get('protocol')!='FG_S3_E_QUALITY_F1_RUN_V1'
            or identity.get('backend')!='external_immutable_course_reader'
            or identity.get('preparation',{}).get('course_stage')!='FOUNDATION_WM'):
        raise ValueError('F2_SOURCE_MUST_BE_F1')
    engineering=identity.get('controlled_small') is True
    if engineering and not allow_engineering:raise ValueError('F2_FOUNDATION_NOT_TRAINED')
    if not engineering and identity.get('training_admitted') is not True:raise ValueError('F2_F1_FORMAL_TRAINING_REQUIRED')
    budget=identity['preparation']['optimizer_updates']
    agent=pickle.loads((node/'agent.pkl').read_bytes())
    counters=pickle.loads((node/'counters.pkl').read_bytes())
    if (type(budget) is not int or budget<=0 or meta['step']!=budget or not meta['protected']
            or agent['counters']['updates']!=budget or counters['optimizer_updates']!=budget
            or counters['consumed_batches']!=budget):raise ValueError('F2_F1_NOT_COMPLETED')
    if int(np.asarray(agent['params']['opt/step/value']))!=budget:raise ValueError('F2_F1_OPTIMIZER_COUNT_MISMATCH')
    from gym_trading_env.research.fg_s3_e_quality_heads import model_config
    if identity['effective_model']['agent']!=plain(model_config(settings,controlled_small=engineering))['agent']:
        raise ValueError('F2_F1_MODEL_ARCHITECTURE_MISMATCH')
    from gym_trading_env.research.fg_s3_e_observation import observation_contract
    from dreamerv3.fg_s3_e_quality_course_contract import REWARD_SCHEMA
    p=identity['preparation']
    if p['observation']!=observation_contract() or p['reward_schema']!=REWARD_SCHEMA:raise ValueError('F2_F1_REWARD_OBSERVATION_CLOCK_MISMATCH')
    proof=dict(protocol='FG_S3_F1_SOURCE_V1',run_identity_sha256=digest(root/'run_identity.json'),
        manifest_sha256=digest(node/'manifest.json'),checkpoint=str(node),run=str(root),
        identity=identity,engineering_only=engineering,foundation_status='FOUNDATION_NOT_TRAINED' if engineering else 'FORMAL_F1_COMPLETED')
    proof['training_evidence']=dict(status='ENGINEERING_ONLY') if engineering else training_evidence(root,identity,agent)
    return proof,agent

def transfer_parameters(agent,source,proof):
    """All validation precedes mutation. Neither source optimizer nor counters load."""
    import jax
    from embodied.jax import internal
    from dreamerv3.fg_s3a_actor import parameter_digests
    fresh=agent.save_exact()
    if any(fresh['counters'].values()) or any(fresh[k] is not None for k in ('pending_sync','pending_outs','pending_mets')):
        raise ValueError('F2_HANDOFF_REQUIRES_FRESH_AGENT')
    keys={k for k in fresh['params'] if not k.startswith('opt/')}
    other={k for k in source['params'] if not k.startswith('opt/')}
    if keys!=other:raise ValueError('F2_HANDOFF_PARAMETER_KEYS_DIFFER')
    for key in sorted(keys):
        src,dst=source['params'][key],fresh['params'][key]
        if src.shape!=dst.shape or src.dtype!=dst.dtype or not np.isfinite(src).all():raise ValueError('F2_HANDOFF_SHAPE_DTYPE_OR_FINITE: '+key)
    before=parameter_digests(fresh['params'])
    # Original Agent sharding conversion; no historical load/counter path.
    copied={k:internal.device_put(source['params'][k].copy(),agent.train_params_sharding[k]) for k in keys}
    with agent.train_lock:
        old=[agent.params[k] for k in keys]
        agent.params.update(copied)
        jax.tree.map(lambda x:x.delete(),old)
    if agent.jaxcfg.enable_policy:agent.sync_policy_from_train()
    after=parameter_digests(agent.params);expected=parameter_digests({k:source['params'][k] for k in keys})
    if any(after[k]!=expected[k] for k in keys) or any(after[k]!=before[k] for k in after if k.startswith('opt/')):
        raise RuntimeError('F2_HANDOFF_BYTE_AUDIT_FAILED')
    return signed(dict(protocol='F1_TO_F2_TRANSFER_RECEIPT',source=proof,engineering_only=proof['engineering_only'],
        foundation_status=proof['foundation_status'],nonopt_keys=sorted(keys),source_nonopt=expected,target_nonopt={k:after[k] for k in sorted(keys)},
        fresh_optimizer={k:after[k] for k in sorted(after) if k.startswith('opt/')},fresh_counters=fresh['counters'],target_seed=int(agent.config.seed),reader_stage='HEADS_VALID',
        same_run_resume=False,training_admitted=False,nonopt_identity_hash=fingerprint(expected)))
