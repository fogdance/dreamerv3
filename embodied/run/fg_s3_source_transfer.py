"""Actual complete Agent transfer and byte-level audit; no BC optimizer import."""
from pathlib import Path
import pickle

import elements
import jax
import numpy as np

from embodied.run.fg_checkpoint import file_hash
from embodied.run.fg_s3_source_admission import prepare_transfer
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import signed


def make_agent(config, output, *, agent_cls=None, device='cpu', bc_lambda=None):
    from dreamerv3.agent import Agent
    from gym_trading_env.research.fg_replay import FIELDS
    obs = {k:elements.Space(dtype, shape) for k,(shape,dtype) in FIELDS.items() if k!='action'}
    act = {'action': elements.Space(np.int32, (), 0, 2)}
    cfg = elements.Config(config)
    jcfg = dict(cfg.jax)
    jcfg.update(platform='cpu' if device=='cpu' else 'cuda', precompile=False,
                profiler=False, prealloc=False, enable_policy=True, transfer_guard=False,
                sync_train_metrics=True)
    extra = {} if bc_lambda is None else dict(bc_lambda=bc_lambda)
    return (agent_cls or Agent)(obs, act, elements.Config(**dict(cfg.agent), **extra,
        logdir=str(output), seed=cfg.seed, jax=jcfg, batch_size=cfg.batch_size,
        batch_length=cfg.batch_length, replay_context=cfg.replay_context,
        report_length=cfg.report_length, replica=0, replicas=1))


def apply_and_audit_transfer(agent, spec):
    schema = {k:(v.shape,str(v.dtype)) for k,v in agent.params.items()}
    state = prepare_transfer(spec, schema)
    if state is None:
        if int(agent.n_updates) or int(agent.n_batches) or int(agent.n_actions):
            raise ValueError('direct_init target is not fresh')
        state = agent.save_exact()
    before = parameter_digests(state['params'])
    prepared = dict(state)
    initialized_policy = state['policy_params'] is None
    if initialized_policy:
        # Legitimate offline sources may disable inference. Build the complete
        # target policy view from the same authenticated train parameters.
        prepared['policy_params'] = {k:state['params'][k] for k in agent.policy_keys}
        prepared['pending_sync'] = None
    else:
        policy = state['policy_params']
        if set(policy) != set(agent.policy_keys) or any(
                (v.shape,str(v.dtype)) != schema[k] or not np.isfinite(v).all()
                for k,v in policy.items()):
            raise ValueError('Source inference policy schema/finite mismatch')
    agent.load_exact(prepared, transfer=True)
    agent.sync_policy_from_train()  # no reset_counter: source counters survive.
    after = parameter_digests(agent.params)
    if before != after:
        raise RuntimeError('Full Agent transfer changed parameter/state bytes')
    saved = agent.save_exact()
    if saved['counters'] != state['counters'] or saved['pending_sync'] is not None:
        raise RuntimeError('Transfer counters or policy synchronization differ')
    policy = parameter_digests(saved['policy_params'])
    if any(policy[k] != after[k] for k in policy):
        raise RuntimeError('Transferred inference policy differs')
    baseline_equality = 'NOT_VERIFIED'
    baseline = spec['origin'].get('baseline')
    if baseline:
        path = Path(baseline['path'])
        if file_hash(path/'manifest.json') != baseline['manifest_sha256']:
            raise ValueError('Baseline changed after admission')
        original = pickle.loads((path/'agent.pkl').read_bytes())
        base_hashes = parameter_digests(original['params'])
        if any(after[k] != base_hashes[k] for k in after if not k.startswith('pol/')):
            raise RuntimeError('BC non-Actor/base joint optimizer equality failed')
        baseline_equality = 'PASS'
    return signed(dict(status='PASS_REAL_MODEL_TRANSFER', devices=[str(x) for x in jax.devices()],
        diagnostic=spec['origin']['diagnostic'],
        transfer_code_sha256=file_hash(__file__),
        source_admission_sha256=spec['content_hash'], kind=spec['kind'],
        source_update_offset=spec['origin']['step'], source_bc_updates=spec['origin'].get('bc_updates',0),
        optimizer_transfer_policy=spec['optimizer_transfer_policy'],
        params_before=before, params_after=after, policy_params=policy,
        joint_opt={k:v for k,v in after.items() if k.startswith('opt/')},
        gate_norm={k:v for k,v in after.items() if k.startswith(('avail_actor_gate/','retnorm/','valnorm/','advnorm/','slowval/'))},
        counters=saved['counters'], baseline_non_actor_optimizer_equality=baseline_equality,
        discarded_pending_outs=state['pending_outs'] is not None,
        discarded_pending_mets=state['pending_mets'] is not None,
        policy_initialized_from_train=initialized_policy,
        policy_synchronized=True, independent_bc_optimizer_imported=False))
