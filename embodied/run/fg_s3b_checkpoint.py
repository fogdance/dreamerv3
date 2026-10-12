"""Distinct S3-B checkpoint protocol, exact typed Optax state and strict resume."""
import json
import os
from pathlib import Path
import pickle
import uuid

import jax
import numpy as np

from dreamerv3.fg_s3a_actor import parameter_digests
from dreamerv3.fg_s3b_bc import LOSS_VERSION
from embodied.run.fg_checkpoint import file_hash
from embodied.run.fg_s3a_checkpoint import code_identity as s3a_code_identity,package_versions
from gym_trading_env.research.fg_replay import check_files,read_manifest,signed,write_json

PROTOCOL = 'FG_S3B_BC_V2'


def code_identity():
    from gym_trading_env.research.fg_s3b_normalization import code_identity as gym_identity
    root = Path(__file__).resolve().parents[2]
    extra = ['dreamerv3/fg_s3b_bc.py','embodied/run/fg_s3b_checkpoint.py',
             'embodied/run/fg_s3b_train.py','scripts/fg_s3b.py']
    result = s3a_code_identity()
    result.update({p:file_hash(root/p) for p in extra})
    result.update({'gym_s3b/'+p:h for p,h in gym_identity().items()})
    return result


def save_bc(directory,bc,sampler,identity):
    if (identity['loss_normalization']['loss_version'] != LOSS_VERSION
            or identity['loss_normalization']['global_mean_weight'] != bc.global_mean_weight):
        raise ValueError('S3-B frozen loss normalization differs from training identity')
    root = Path(directory); root.mkdir(parents=True,exist_ok=True)
    dest = root/f'bc_{bc.bc_updates:012d}_{bc.bc_attempts:012d}'
    if dest.exists(): raise FileExistsError('S3-B checkpoint already exists')
    pending = root/('.pending_'+uuid.uuid4().hex); pending.mkdir()
    audit = bc.freeze_audit(require_actor_change=bc.bc_updates > 0)
    state = dict(agent=bc.agent.save_exact(),loss_version=LOSS_VERSION,
        global_mean_weight=bc.global_mean_weight,
        s3b_bc_opt=jax.tree.map(lambda a:np.array(jax.device_get(a),copy=True),bc.s3b_bc_opt),
        bc_updates=bc.bc_updates,bc_attempts=bc.bc_attempts,skipped_batches=bc.skipped_batches,
        sampler=sampler.save(),source_digests=bc.source_digests,source_shapes=bc.source_shapes)
    with (pending/'state.pkl').open('wb') as f:
        pickle.dump(state,f,protocol=5); f.flush(); os.fsync(f.fileno())
    write_json(pending/'freeze_audit.json',audit)
    write_json(pending/'manifest.json',signed(dict(protocol=PROTOCOL,identity=identity,
        bc_updates=bc.bc_updates,bc_attempts=bc.bc_attempts,skipped_batches=bc.skipped_batches,
        files={p.name:file_hash(p) for p in pending.iterdir()})))
    (pending/'done').write_text('')
    pending.rename(dest)
    pointer = root/'.latest.pending'; pointer.write_text(dest.name+'\n'); os.replace(pointer,root/'latest')
    return dest


def validate_bc(path,identity=None):
    path = Path(path); m = read_manifest(path/'manifest.json')
    if (not (path/'done').is_file() or m['protocol'] != PROTOCOL
            or (identity is not None and identity != m['identity'])
            or m['identity']['code_sha256'] != code_identity()
            or m['identity']['packages'] != package_versions()):
        raise ValueError('Foreign/incomplete S3-B checkpoint identity')
    if {p.name for p in path.iterdir()} != set(m['files'])|{'manifest.json','done'}:
        raise ValueError('S3-B checkpoint file set mismatch')
    check_files({str(path/p):h for p,h in m['files'].items()})
    return m


def audit_bc(path,identity=None):
    m = validate_bc(path,identity)
    state = pickle.loads((Path(path)/'state.pkl').read_bytes())
    proof = json.loads((Path(path)/'freeze_audit.json').read_text())
    norm = m['identity']['loss_normalization']
    if (state['loss_version'] != LOSS_VERSION or norm['loss_version'] != LOSS_VERSION
            or proof['loss_version'] != LOSS_VERSION
            or state['global_mean_weight'] != norm['global_mean_weight']
            or proof['global_mean_weight'] != norm['global_mean_weight']
            or norm['global_mean_weight_hex'] != float(norm['global_mean_weight']).hex()
            or not np.isfinite(norm['global_mean_weight']) or norm['global_mean_weight'] <= 0):
        raise ValueError('S3-B saved global loss normalization differs')
    params = state['agent']['params']
    before,after = state['source_digests'],parameter_digests(params)
    changed = sorted(k for k in before if before[k] != after.get(k))
    if (set(before) != set(after) or any(not k.startswith('pol/') for k in changed)
            or (m['bc_updates'] > 0 and not changed) or proof['source_digests'] != before
            or proof['current_digests'] != after or proof['changed_keys'] != changed
            or any((v.shape,str(v.dtype)) != state['source_shapes'][k] for k,v in params.items())):
        raise ValueError('S3-B frozen source/parameter schema proof differs')
    for name in ('bc_updates','bc_attempts','skipped_batches'):
        if state[name] != m[name] or proof[name] != state[name]:
            raise ValueError('S3-B independent counters differ')
    if state['bc_attempts'] != state['bc_updates']+state['skipped_batches']:
        raise ValueError('S3-B attempt/update/skip accounting differs')
    if any(not np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(state['s3b_bc_opt'])):
        raise ValueError('S3-B optimizer contains nonfinite state')
    if (proof['original_updates'] != state['agent']['counters']['updates']
            or proof['original_batches'] != state['agent']['counters']['batches']):
        raise ValueError('S3-B original counters differ')
    policy = state['agent']['policy_params']
    policy_digests = parameter_digests(policy) if policy is not None else None
    if (policy is None or state['agent']['pending_sync'] is not None
            or any(policy_digests[k] != after[k] for k in policy)):
        raise ValueError('S3-B inference is not using the updated parameters')
    return dict(m,freeze_audit=proof)


def restore_bc(path,bc,sampler,identity):
    m = audit_bc(path,identity)
    if m['identity']['loss_normalization']['global_mean_weight'] != bc.global_mean_weight:
        raise ValueError('S3-B restore frozen global mean differs')
    state = pickle.loads((Path(path)/'state.pkl').read_bytes())
    if state['source_digests'] != bc.source_digests or state['source_shapes'] != bc.source_shapes:
        raise ValueError('S3-B source checkpoint fingerprint/schema differs')
    if jax.tree.structure(state['s3b_bc_opt']) != jax.tree.structure(bc.s3b_bc_opt):
        raise ValueError('S3-B optimizer structure differs')
    for saved,expected in zip(jax.tree.leaves(state['s3b_bc_opt']),jax.tree.leaves(bc.s3b_bc_opt)):
        if saved.shape != expected.shape or saved.dtype != expected.dtype or not np.isfinite(saved).all():
            raise ValueError('S3-B optimizer shape/dtype/finite contract differs')
    if any(state['agent']['counters'][k] != v for k,v in bc.source_counters.items()):
        raise ValueError('S3-B original joint counters changed')
    sampler.load(state['sampler'])
    bc.agent.load_exact(state['agent'])
    bc.agent.sync_policy_from_train()
    bc.s3b_bc_opt = jax.tree.map(jax.device_put,state['s3b_bc_opt'])
    bc.bc_updates,bc.bc_attempts,bc.skipped_batches = state['bc_updates'],state['bc_attempts'],state['skipped_batches']
    bc.frozen_refs = {k:v for k,v in bc.agent.params.items() if k not in bc.actor_keys}
    bc.freeze_audit(require_actor_change=bc.bc_updates > 0)
    return m
