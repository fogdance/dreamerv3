"""Explicit F1/F2 stage parameter handoff, distinct from exact resume."""
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
    from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
    freeze=source_freeze_evidence(root,identity,agent,WM_GROUPS)
    proof=dict(protocol='FG_S3_F1_SOURCE_V2',run_identity_sha256=digest(root/'run_identity.json'),
        manifest_sha256=digest(node/'manifest.json'),checkpoint=str(node),run=str(root),
        identity=identity,source_freeze=freeze,engineering_only=engineering,foundation_status='FOUNDATION_NOT_TRAINED' if engineering else 'FORMAL_F1_COMPLETED')
    proof['training_evidence']=dict(status='ENGINEERING_ONLY') if engineering else training_evidence(root,identity,agent)
    return proof,agent

def transfer_parameters(agent,source,proof,*,target_stage='QUALITY_HEADS'):
    """All validation precedes mutation. Neither source optimizer nor counters load."""
    import jax
    from embodied.jax import internal
    from dreamerv3.fg_s3a_actor import parameter_digests
    if target_stage not in ('QUALITY_HEADS','ACTOR_SUPERVISED','ACTOR_ONLINE'):raise ValueError('QUALITY_HANDOFF_TARGET_STAGE')
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
    return signed(dict(protocol='F1_TO_F2_TRANSFER_RECEIPT' if target_stage=='QUALITY_HEADS' else 'F3_TO_ONLINE_TRANSFER_RECEIPT' if target_stage=='ACTOR_ONLINE' else 'F1_OR_F2_TO_F3_TRANSFER_RECEIPT',source=proof,engineering_only=proof['engineering_only'],
        foundation_status=proof['foundation_status'],nonopt_keys=sorted(keys),source_nonopt=expected,target_nonopt={k:after[k] for k in sorted(keys)},
        fresh_optimizer={k:after[k] for k in sorted(after) if k.startswith('opt/')},fresh_counters=fresh['counters'],target_seed=int(agent.config.seed),reader_stage='HEADS_VALID' if target_stage=='QUALITY_HEADS' else 'ACTOR_STRICT',
        same_run_resume=False,training_admitted=False,nonopt_identity_hash=fingerprint(expected)))


def source_freeze_evidence(root,identity,final,groups):
    """Inspect actual initial/final source bytes, including protected slots."""
    initial=Path(root)/'ckpt/step_000000000000';validate_bundle(initial,identity)
    first=pickle.loads((initial/'agent.pkl').read_bytes())
    if first['counters']['updates']!=0:raise ValueError('F3_SOURCE_INITIAL_COUNTER_INVALID')
    a,b=first['params'],final['params']
    if a.keys()!=b.keys() or any(a[k].shape!=b[k].shape or a[k].dtype!=b[k].dtype
            or not np.isfinite(b[k]).all() or not np.isfinite(a[k]).all() for k in a):
        raise ValueError('F3_SOURCE_FULL_STATE_SCHEMA_OR_FINITE')
    # Use the execution schema's exact optimizer mapping, including its
    # configured shared counter. A suffix named "count" is not a clock permit.
    from types import SimpleNamespace
    import elements
    from dreamerv3.fg_s3_e_quality_foundation import state_owners
    config=identity['effective_model'] if 'effective_model' in identity else identity['preparation']['effective_model']
    model=SimpleNamespace(config=elements.Config(config['agent']))
    owners=state_owners(model,a,groups)
    frozen=[];changed=[]
    for k in a:
        if owners[k]=='CLOCK':continue
        if owners[k] is not None:
            if a[k].tobytes()!=b[k].tobytes() and not k.startswith('opt/'):changed.append(k)
        else:
            if a[k].tobytes()!=b[k].tobytes():raise ValueError('F3_SOURCE_FROZEN_STATE_CHANGED: '+k)
            frozen.append(k)
    if not changed:raise ValueError('F3_SOURCE_PARAMETERS_NOT_TRAINED')
    return dict(protocol='FG_S3_SOURCE_FREEZE_EVIDENCE_V2',
                initial_manifest_sha256=digest(initial/'manifest.json'),changed_parameters=changed,
                changed_groups={p:sum(k.startswith(p) for k in changed) for p in groups},
                shared_clocks=list(model.f1_shared_clocks),
                frozen_keys=frozen,all_frozen_bytes_equal=True)


def inspect_stage(settings,*,allow_engineering=False):
    """F1 or F2 source for a new F3 run; never a same-run resume."""
    from gym_trading_env.research.fg_s3_e_quality_actor import model_config
    from gym_trading_env.research.fg_s3_e_observation import observation_contract
    from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
    from dreamerv3.fg_s3_e_quality_heads import GROUPS as HEADS,execution_contract
    stage=settings['source_stage'];root=Path(settings['source_run']);node=Path(settings['source_checkpoint'])
    if stage not in ('F1','F2') or not node.is_relative_to(root/'ckpt'):raise ValueError('F3_SOURCE_STAGE_OR_NODE_INVALID')
    if stage=='F1':
        legacy={**settings,'source_f1_run':str(root),'source_f1_checkpoint':str(node)}
        proof,agent=inspect_f1(legacy,allow_engineering=allow_engineering)
        proof={**proof,'source_stage':'F1'}
        return proof,agent
    identity=read_manifest(root/'run_identity.json')['identity'];meta=validate_bundle(node,identity)
    if identity.get('protocol')!='FG_S3_E_QUALITY_F2_RUN_V1' or identity.get('backend')!='external_immutable_course_reader':
        raise ValueError('F3_SOURCE_MUST_BE_F2')
    engineering=identity.get('controlled_small') is True
    if engineering and not allow_engineering:raise ValueError('F3_FOUNDATION_NOT_TRAINED')
    if not engineering and identity.get('training_admitted') is not True:raise ValueError('F3_FORMAL_SOURCE_REQUIRED')
    prep=identity['preparation'];s=prep['settings']
    if (prep.get('protocol')!='FG_S3_E_QUALITY_HEADS_V1' or s['stage']!='QUALITY_HEADS'
            or prep['execution_contract']!=execution_contract() or prep['observation']!=observation_contract()
            or prep['effective_model']['agent']!=plain(model_config(settings,controlled_small=engineering))['agent']):
        raise ValueError('F3_SOURCE_MODEL_OBSERVATION_OR_STAGE_MISMATCH')
    upstream,_=inspect_f1(s,allow_engineering=allow_engineering)
    if engineering!=upstream['engineering_only'] or any(upstream[k]!=prep['source_f1'].get(k)
            for k in ('protocol','manifest_sha256','run_identity_sha256','source_freeze','training_evidence')):
        raise ValueError('F3_SOURCE_UPSTREAM_LINEAGE_MISMATCH')
    agent=pickle.loads((node/'agent.pkl').read_bytes());c=pickle.loads((node/'counters.pkl').read_bytes())
    budget=s['optimizer_updates'];consumed=c['consumed_batches']
    if (type(budget) is not int or budget<=0 or not meta['protected'] or meta['step']!=consumed
            or c['optimizer_updates']!=budget or agent['counters']['updates']!=budget
            or int(np.asarray(agent['params']['opt/step/value']))!=budget):raise ValueError('F3_SOURCE_NOT_COMPLETED')
    import json
    rows=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    if len(rows)!=consumed:raise ValueError('F3_SOURCE_UPDATE_LEDGER_INCOMPLETE')
    count=0
    for i,row in enumerate(rows,1):
        m=row['metrics'];updated=m['f2/updated']
        if type(updated) is not bool:raise ValueError('F3_SOURCE_UPDATE_LEDGER_INVALID')
        count+=int(updated)
        if (row['consumed_batches']!=i or row['local_update']!=count or row['agent_update']!=count
                or m['opt/updates']!=count or m['f2/finite'] is not True
                or not np.isfinite(m['opt/loss']) or not np.isfinite(m['opt/grad_norm'])):
            raise ValueError('F3_SOURCE_UPDATE_LEDGER_INVALID')
    if count!=budget:raise ValueError('F3_SOURCE_UPDATE_COUNT_MISMATCH')
    proof=dict(protocol='FG_S3_F2_SOURCE_V2',source_stage='F2',run=str(root),checkpoint=str(node),
        run_identity_sha256=digest(root/'run_identity.json'),manifest_sha256=digest(node/'manifest.json'),
        identity=identity,engineering_only=engineering,foundation_status='FOUNDATION_NOT_TRAINED' if engineering else 'FORMAL_F2_COMPLETED',
        upstream_proof=upstream,training_evidence=dict(consumed_batches=consumed,verified_updates=count,update_log_sha256=digest(root/'updates.jsonl')),
        source_freeze=source_freeze_evidence(root,identity,agent,HEADS))
    return proof,agent
