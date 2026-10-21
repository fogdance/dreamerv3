"""Bounded F4 admission state and read-only historical Reader forks, no trainer."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gym_trading_env.envs.entry_contract import fingerprint

PROTOCOL='FG_S3_E_QUALITY_RL_ADMISSION_STATE_V1'
POLICY_FIELDS={'warmup_env_steps','calibration_batches','calibration_horizon','calibration_seed'}
STAGES={'DISABLED','DEV_NUMERIC_PROBE','WARMUP','WM_CALIBRATING','RL_PROBING',
    'RL_BLOCKED','AWAIT_OPERATOR_REVIEW','RL_ACTIVE','FINAL_VERIFY','COMPLETE_VERIFIED'}


def validate_rl_policy(policy,spec):
    if policy is None and spec['execution_mode']!='ONLINE_RL_CONTROLLED':return None
    if (type(policy) is not dict or set(policy)!=POLICY_FIELDS
            or any(type(v) is not int or v<0 for v in policy.values())
            or not 0<policy['warmup_env_steps']<spec['env_steps']
            or policy['warmup_env_steps']%spec['collect_every']
            or policy['calibration_batches']<1 or policy['calibration_horizon']<1):
        raise ValueError('ONLINE_EXPLICIT_BOUNDED_RL_ADMISSION_POLICY_REQUIRED')
    if spec['execution_mode']!='ONLINE_RL_CONTROLLED':
        raise ValueError('ONLINE_RL_POLICY_ONLY_FOR_CONTROLLED_RL')
    return copy.deepcopy(policy)


class RLAdmissionState:
    def __init__(self,run_identity,policy,*,numeric_probe=False):
        self.identity=run_identity;self.policy=copy.deepcopy(policy)
        stage='DEV_NUMERIC_PROBE' if numeric_probe else 'DISABLED' if policy is None else 'WARMUP'
        self.state=dict(protocol=PROTOCOL,run_identity=run_identity,policy=self.policy,stage=stage,
            attempt=0,blocked_count=0,reason=None,technical=None,last_review_sha256=None)

    def transition(self,stage,*,reason=None):
        allowed={
            'WARMUP':{'WM_CALIBRATING','RL_BLOCKED'},
            'WM_CALIBRATING':{'WM_CALIBRATING','RL_PROBING','RL_BLOCKED'},
            'RL_PROBING':{'WM_CALIBRATING','AWAIT_OPERATOR_REVIEW','RL_BLOCKED'},
            'RL_BLOCKED':{'WM_CALIBRATING','RL_BLOCKED'},
            'AWAIT_OPERATOR_REVIEW':{'RL_ACTIVE','RL_BLOCKED'},
            'RL_ACTIVE':{'RL_ACTIVE','RL_BLOCKED','FINAL_VERIFY'},
            'FINAL_VERIFY':{'FINAL_VERIFY','COMPLETE_VERIFIED','RL_BLOCKED'},
            'COMPLETE_VERIFIED':set(), 'DISABLED':set(), 'DEV_NUMERIC_PROBE':set()}
        current=self.state['stage']
        if stage not in allowed[current]:
            raise ValueError('ONLINE_ILLEGAL_RL_GATE_TRANSITION: '+current+' -> '+stage)
        self.state['stage']=stage;self.state['reason']=reason
        if stage=='RL_BLOCKED':self.state['blocked_count']+=1

    def save(self):return copy.deepcopy(self.state)

    def validate_snapshot(self,state=None):
        state=self.state if state is None else state
        if (type(state) is not dict or set(state)!=set(self.state) or state.get('protocol')!=PROTOCOL
                or state.get('run_identity')!=self.identity or state.get('policy')!=self.policy
                or state.get('stage') not in STAGES
                or any(type(state.get(k)) is not int or state[k]<0 for k in ('attempt','blocked_count'))):
            raise ValueError('ONLINE_RL_LIFECYCLE_SNAPSHOT_INVALID')
        if state['stage'] in ('AWAIT_OPERATOR_REVIEW','RL_ACTIVE','FINAL_VERIFY','COMPLETE_VERIFIED'):
            if type(state['technical']) is not dict or not state['technical'].get('evidence_hash'):
                raise ValueError('ONLINE_RL_TECHNICAL_EVIDENCE_REQUIRED')

    def load(self,state):self.validate_snapshot(state);self.state=copy.deepcopy(state)


def fork_history(history,*,seed):
    """Open the same WP03 source descriptors with independent Reader/scheduler RNG."""
    from gym_trading_env.research.fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    natural=history.role_streams['NATURAL'].stream
    sources=[dict(corpus_id=name,admission=reader.admission,index_path=reader.index_path)
        for name,reader in natural.readers.items()]
    identity=history.identity
    specs={row['corpus_id']:row['sampling_spec'] for row in natural.identity['sources']}
    weights={row['corpus_id']:row['weight'] for row in natural.identity['sources']}
    aux=None
    if 'AUXILIARY' in history.role_streams:
        aux={row['corpus_id']:row['sampling_spec']
            for row in history.role_streams['AUXILIARY'].stream.identity['sources']}
    return MultiCorpusF3Stream.open(sources,settings=identity['settings'],
        per_source_sampling_spec=specs,source_mix_weights=weights,seed=seed,
        aux_per_source_sampling_spec=aux,cache_bytes=1<<20)


def capture_live(run,directory):
    """Reuse the FullCheckpoint Replay snapshot, without restoring or consuming it."""
    import random
    import numpy as np
    from .fg_checkpoint import save_replay,host
    from .fg_s3_e_quality_online_probe import state_digest
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    state=dict(agent=run.agent.save_exact(),runtime=run.runtime.save(),collector=run.collector.save(),
        driver=run.collector.driver.save(),publisher=None if run.publisher is None else run.publisher.save(),
        source=None if run.source is None else run.source.save_full(),counters=run.counters.save(),
        phase=run.phase.save(),gates=run.gates.save(),source_exposures=run.source_exposures.save(),
        account=run.collector.env.export_state(),account_verification=run.account_verification.save(),
        lifecycle=run.rl_lifecycle.save(),market_step=int(run.step),transaction=int(run.transaction),
        consumed_step=int(run.consumed_step),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        replay=save_replay(run.collector.replay,directory/'replay'))
    state['replay_files']={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
        for p in sorted((directory/'replay').iterdir())}
    return {name:state_digest(host(value)) for name,value in state.items()}


class LiveStateChanged(RuntimeError):
    """Keep the pre-probe CP; never publish an unproven live mutation."""
