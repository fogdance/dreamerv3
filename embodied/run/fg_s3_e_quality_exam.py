"""Read-only stage dispatch over original Checkpoint/source validators."""
from pathlib import Path
import copy
import pickle

import numpy as np

from .fg_checkpoint import validate_bundle
from .fg_s3_e_quality_online import inspect_f3
from gym_trading_env.research.fg_replay import read_manifest,digest,signed
from gym_trading_env.research.fg_s3_e_quality_course_preflight import _local_path


def _f3_spec(root,node):
    # Only an input adapter for the existing inspector; never an online Run.
    from dreamerv3.fg_s3_e_quality_online import PROTOCOL
    return dict(protocol=PROTOCOL,source_stage='ACTOR_SUPERVISED',execution_mode='SHADOW',
        fold='TRAIN',source_run=str(root),source_checkpoint=str(node),run_dir='/tmp/wp08-inspection-unused',
        model_seed=0,env_seed=0,replay_seed=0,env_steps=1,collect_every=1,updates_per_collect=0,
        max_consumed_batches=0,save_every=1,replay_capacity=512,exploration_rate=0.,history_ratio=.5)


def inspect_exam_actor(kind,run,checkpoint,*,engineering=False):
    root=_local_path(run);node=_local_path(checkpoint)
    if not node.is_relative_to(root/'ckpt'):raise ValueError('EXAM_CHECKPOINT_OUTSIDE_SOURCE_RUN')
    record=read_manifest(root/'run_identity.json');ident=record['identity']
    meta=validate_bundle(node,ident)
    if kind=='F3':
        proof,state=inspect_f3(_f3_spec(root,node),allow_engineering=engineering)
        settings=copy.deepcopy(ident['preparation']['settings']);mode=None
    elif kind=='F4':
        from dreamerv3.fg_s3_e_quality_online import execution_contract
        if ident.get('protocol')!='FG_S3_E_QUALITY_ONLINE_RUN_V1':raise ValueError('EXAM_F4_SOURCE_REQUIRED')
        mode=ident['spec']['execution_mode']
        if ident['execution_contract']!=execution_contract(mode):raise ValueError('EXAM_F4_EXECUTION_CONTRACT_MISMATCH')
        proof,_=inspect_f3(ident['spec'],allow_engineering=engineering)
        if (proof['manifest_sha256']!=ident['source_manifest']
                or proof['run_identity_sha256']!=ident['source_run_identity']
                or proof['engineering_only']!=ident['controlled_small']):
            raise ValueError('EXAM_F4_SOURCE_LINEAGE_MISMATCH')
        if mode!='SHADOW' and not ident['controlled_small'] and not ident['training_admitted']:
            raise ValueError('EXAM_F4_SOURCE_NOT_ADMITTED')
        state=pickle.loads((node/'agent.pkl').read_bytes())
        if not state.get('params') or any(not np.isfinite(v).all() for v in state['params'].values()):
            raise ValueError('EXAM_F4_NONFINITE_MODEL_STATE')
        counters=pickle.loads((node/'counters.pkl').read_bytes())
        market=int(pickle.loads((node/'market_step.pkl').read_bytes()))
        if (counters['numerical_rejections'] or state['counters']['updates']!=counters['optimizer_updates']
                or int(np.asarray(state['params']['opt/step/value']))!=counters['optimizer_updates']
                or not 0<=market<=ident['spec']['env_steps']
                or mode=='SHADOW' and counters['optimizer_updates']!=0):
            raise ValueError('EXAM_F4_COUNTER_OR_REJECTION_MISMATCH')
        # Match only the committed prefix for this selected node; later source
        # updates are not part of the selected immutable model.
        import json
        logs=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()] if (root/'updates.jsonl').exists() else []
        consumed=counters['consumed_batches']
        if len(logs)<consumed:raise ValueError('EXAM_F4_COMMITTED_LEDGER_MISSING')
        updated=0
        for i,row in enumerate(logs[:consumed],1):
            metrics=row.get('metrics',{})
            if (row.get('consumed_batches')!=i or metrics.get('f3/finite') is not True
                    or type(metrics.get('f3/updated')) is not bool):
                raise ValueError('EXAM_F4_COMMITTED_LEDGER_INVALID')
            updated+=int(metrics['f3/updated'])
            if row.get('agent_update')!=updated:raise ValueError('EXAM_F4_COMMITTED_LEDGER_INVALID')
        if updated!=counters['optimizer_updates']:raise ValueError('EXAM_F4_COMMITTED_LEDGER_COUNT')
        settings=copy.deepcopy(proof['identity']['preparation']['settings'])
        settings['model_init']['seed']=ident['spec']['model_seed']
    else:raise ValueError('EXAM_EXPLICIT_F3_F4_SOURCE_KIND_REQUIRED')
    if not engineering and proof['engineering_only']:raise ValueError('EXAM_SYNTHETIC_MODEL_IS_NOT_FORMAL')
    receipt=signed(dict(protocol='FG_S3_E_QUALITY_FROZEN_ACTOR_SOURCE_V1',kind=kind,
        run=str(root),checkpoint=str(node),run_identity_sha256=digest(root/'run_identity.json'),
        checkpoint_manifest_sha256=digest(node/'manifest.json'),source_identity=ident,source_proof=proof,
        engineering_only=proof['engineering_only'],execution_mode=mode,
        member_hashes=meta['files'],course=proof['identity']['preparation']['course']))
    return receipt,state,settings
