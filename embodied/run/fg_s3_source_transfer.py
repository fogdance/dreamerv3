"""Actual complete Agent transfer and byte-level audit; no BC optimizer import."""
from pathlib import Path
import pickle

import elements
import jax
import numpy as np

from embodied.run.fg_checkpoint import file_hash, plain
from embodied.run.fg_s3_source_admission import prepare_transfer
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import signed
from gym_trading_env.envs.entry_contract import fingerprint


def make_agent(config, output, *, agent_cls=None, device='cpu', bc_lambda=None,
               extra_agent_config=None):
    from dreamerv3.agent import Agent
    from gym_trading_env.research.fg_replay import FIELDS
    obs = {k:elements.Space(dtype, shape) for k,(shape,dtype) in FIELDS.items() if k!='action'}
    act = {'action': elements.Space(np.int32, (), 0, 2)}
    cfg = elements.Config(config)
    if device not in ('cpu', 'gpu'):
        raise ValueError('Unknown model device')
    jcfg = dict(cfg.jax)
    # Explicit CUDA avoids the generic gpu alias also initializing ROCm.
    jcfg.update(platform='cpu' if device=='cpu' else 'cuda', precompile=False,
                profiler=False, prealloc=False, enable_policy=True, transfer_guard=False,
                sync_train_metrics=True)
    extra = {} if bc_lambda is None else dict(bc_lambda=bc_lambda)
    extra.update(extra_agent_config or {})
    if set(extra).intersection(cfg.agent):
        raise ValueError('Training extension must not overwrite admitted Agent config')
    return (agent_cls or Agent)(obs, act, elements.Config(**dict(cfg.agent), **extra,
        logdir=str(output), seed=cfg.seed, jax=jcfg, batch_size=cfg.batch_size,
        batch_length=cfg.batch_length, replay_context=cfg.replay_context,
        report_length=cfg.report_length, replica=0, replicas=1))


def apply_and_audit_transfer(agent, spec):
    expected = spec['target_config']['agent']
    if {k:plain(agent.config[k]) for k in expected} != expected:
        raise ValueError('Target model network/optimizer/loss config differs from SourceAdmission')
    if agent.config.avail_threshold != spec['gate_policy']['classification_threshold']:
        raise ValueError('Target model threshold differs from SourceAdmission')
    schema = {k:(v.shape,str(v.dtype)) for k,v in agent.params.items()}
    state = prepare_transfer(spec, schema)
    if spec['kind'] == 'warm_start':
        return _apply_parameter_warm_start(agent, spec, state)
    if state is None:
        if int(agent.n_updates) or int(agent.n_batches) or int(agent.n_actions):
            raise ValueError('direct_init target is not fresh')
        state = agent.save_exact()
    before = parameter_digests(state['params'])
    original = spec.get('source_param_digests', before)
    changed = [k for k in before if before[k] != original[k]]
    allowed = ['avail_actor_gate/value'] if spec['gate_transfer']['reset_required'] else []
    if any(k not in allowed for k in changed):
        raise RuntimeError('Transfer changed parameters outside the admitted Gate reset')
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
        baseline_state = pickle.loads((path/'agent.pkl').read_bytes())
        base_hashes = parameter_digests(baseline_state['params'])
        gate_reset = spec['gate_transfer']['reset_required'] or bool(
            spec['origin'].get('inherited_admission', {}).get('gate_transfer', {}).get('reset_required'))
        if gate_reset:
            key = 'avail_actor_gate/value'
            base_hashes[key] = parameter_digests({key:np.zeros_like(baseline_state['params'][key])})[key]
        if any(after[k] != base_hashes[k] for k in after if not k.startswith('pol/')):
            raise RuntimeError('BC non-Actor/base joint optimizer equality failed')
        baseline_equality = 'PASS'
    return signed(dict(status='PASS_REAL_MODEL_TRANSFER', devices=[str(x) for x in jax.devices()],
        diagnostic=spec['origin']['diagnostic'],
        transfer_code_sha256=file_hash(__file__),
        source_admission_sha256=spec['content_hash'], kind=spec['kind'],
        gate_transfer=spec['gate_transfer'],
        target_config_sha256=fingerprint(spec['target_config']),
        original_source_param_digests=original, gate_reset_changed_keys=changed,
        source_update_offset=spec['origin']['step'], source_bc_updates=spec['origin'].get('bc_updates',0),
        source_checkpoint_step=spec['origin'].get('checkpoint_step',spec['origin']['step']),
        source_checkpoint_step_unit=spec['origin'].get('checkpoint_step_unit','gradient_updates'),
        source_environment_steps=spec['origin'].get('environment_steps'),
        optimizer_transfer_policy=spec['optimizer_transfer_policy'],
        params_before=before, params_after=after, policy_params=policy,
        joint_opt={k:v for k,v in after.items() if k.startswith('opt/')},
        gate_norm={k:v for k,v in after.items() if k.startswith(('avail_actor_gate/','retnorm/','valnorm/','advnorm/','slowval/'))},
        counters=saved['counters'], baseline_non_actor_optimizer_equality=baseline_equality,
        discarded_pending_outs=state['pending_outs'] is not None,
        discarded_pending_mets=state['pending_mets'] is not None,
        policy_initialized_from_train=initialized_policy,
        policy_synchronized=True, independent_bc_optimizer_imported=False))


def _apply_parameter_warm_start(agent, spec, source):
    """New economic semantics: copy only authenticated NN weights into fresh state."""
    groups = ('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'avail/', 'pol/', 'val/')
    fresh = agent.save_exact()
    if any(fresh['counters'].values()):
        raise ValueError('Parameter warm-start requires a fresh target Agent')
    initialized = parameter_digests(fresh['params'])
    keys = {k for k in fresh['params'] if k.startswith(groups)}
    if keys != {k for k in source['params'] if k.startswith(groups)} or any(
            not any(k.startswith(prefix) for k in keys) for prefix in groups):
        raise ValueError('Warm-start network parameter groups differ')
    fresh['params'].update({k: source['params'][k] for k in keys})
    # The new target tracks its copied Critic from its first update.
    slow = {k for k in fresh['params'] if k.startswith('slowval/')}
    fresh['params'].update({k: fresh['params']['val/'+k[len('slowval/'):]] for k in slow})
    fresh['policy_params'] = {k:fresh['params'][k] for k in agent.policy_keys}
    fresh.update(pending_sync=None, pending_outs=None, pending_mets=None)
    agent.load_exact(fresh, transfer=True)
    agent.sync_policy_from_train()
    after = parameter_digests(agent.params)
    copied = parameter_digests({k:source['params'][k] for k in keys})
    if any(after[k] != v for k,v in copied.items()):
        raise RuntimeError('Warm-start altered inherited NN parameter bytes')
    if any(after[k] != initialized[k] for k in after if k not in keys | slow):
        raise RuntimeError('Warm-start changed fresh optimizer/Normalizer/Gate state')
    saved = agent.save_exact()
    if any(saved['counters'].values()) or agent.get_avail_actor_enabled():
        raise RuntimeError('Warm-start inherited counters or Gate activation')
    return signed(dict(status='PASS_REAL_PARAMETER_WARM_START', kind='warm_start',
        devices=[str(d) for d in jax.devices()], source_admission_sha256=spec['content_hash'],
        clock_protocol=spec['clock_protocol'], optimizer_transfer_policy=spec['optimizer_transfer_policy'],
        source_model_updates=spec['origin']['step'],
        source_environment_steps=spec['origin'].get('environment_steps'), source_update_offset=0,
        network_parameter_keys=sorted(keys), inherited_nn_digests=copied,
        fresh_state_digests={k:after[k] for k in after if k not in keys | slow},
        params_before=initialized, params_after=after, counters=saved['counters'],
        slow_critic_initialized_from_critic=True, policy_synchronized=True,
        independent_bc_optimizer_imported=False))
