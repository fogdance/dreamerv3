"""Online lifecycle hooks; original Agent/optimizer/FullCheckpoint are owners."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .fg_checkpoint import validate_bundle
from gym_trading_env.research.fg_replay import read_manifest, digest, signed


def inspect_f3(spec, *, allow_engineering=False):
    from dreamerv3.fg_s3_e_quality_online import validate_execution
    from dreamerv3.fg_s3_e_quality_actor import PROTOCOL, execution_contract
    from gym_trading_env.research.fg_s3_e_observation import observation_contract
    from .fg_s3_e_quality_transfer import inspect_stage, source_freeze_evidence
    s = validate_execution(spec)
    root, node = Path(s['source_run']), Path(s['source_checkpoint'])
    identity = read_manifest(root/'run_identity.json')['identity']
    meta = validate_bundle(node, identity)
    p = identity.get('preparation', {})
    if (identity.get('protocol') != 'FG_S3_E_QUALITY_F3_RUN_V1'
            or identity.get('backend') != 'external_immutable_course_reader'
            or p.get('protocol') != PROTOCOL or p.get('execution_contract') != execution_contract()
            or p.get('observation') != observation_contract()):
        raise ValueError('ONLINE_COMPLETED_F3_SOURCE_REQUIRED')
    engineering = identity.get('controlled_small') is True
    if engineering and not allow_engineering:
        raise ValueError('ONLINE_SYNTHETIC_SOURCE_NOT_FORMAL')
    if not engineering and identity.get('training_admitted') is not True:
        raise ValueError('ONLINE_AUTHENTICATED_F3_TRAINING_REQUIRED')
    upstream, _ = inspect_stage(p['settings'], allow_engineering=engineering)
    if engineering != upstream['engineering_only'] or any(
            upstream[k] != p['source_model'][k] for k in ('manifest_sha256','run_identity_sha256')):
        raise ValueError('ONLINE_F3_UPSTREAM_LINEAGE_MISMATCH')
    state = pickle.loads((node/'agent.pkl').read_bytes())
    c = pickle.loads((node/'counters.pkl').read_bytes())
    budget = p['settings']['optimizer_updates']
    if (not meta['protected'] or meta['step'] != c['consumed_batches']
            or c['optimizer_updates'] != budget or c['numerical_rejections']
            or state['counters']['updates'] != budget
            or int(np.asarray(state['params']['opt/step/value'])) != budget):
        raise ValueError('ONLINE_F3_SOURCE_NOT_COMPLETED')
    rows = [json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    if len(rows) != c['consumed_batches']:
        raise ValueError('ONLINE_F3_LEDGER_INCOMPLETE')
    count = 0
    for i, row in enumerate(rows, 1):
        m = row['metrics']
        if type(m['f3/updated']) is not bool:
            raise ValueError('ONLINE_F3_LEDGER_INVALID')
        count += int(m['f3/updated'])
        if (row['consumed_batches'] != i or row['local_update'] != count or row['agent_update'] != count
                or m['opt/updates'] != count or m['f3/finite'] is not True
                or not np.isfinite(m['opt/loss']) or not np.isfinite(m['opt/grad_norm'])):
            raise ValueError('ONLINE_F3_LEDGER_INVALID')
    if count != budget:
        raise ValueError('ONLINE_F3_LEDGER_COUNT_MISMATCH')
    proof = dict(protocol='FG_S3_F3_ONLINE_SOURCE_V1', identity=identity,
        manifest_sha256=digest(node/'manifest.json'), run_identity_sha256=digest(root/'run_identity.json'),
        checkpoint=str(node), run=str(root), engineering_only=engineering,
        foundation_status='FOUNDATION_NOT_TRAINED' if engineering else 'FORMAL_F3_COMPLETED',
        source_freeze=source_freeze_evidence(root, identity, state, ('quality_actor/',)),
        verified_updates=count, update_log_sha256=digest(root/'updates.jsonl'))
    return signed(proof), state
