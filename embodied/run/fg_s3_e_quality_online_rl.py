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


def perform_admission(run):
    """Called by OnlineRun at its explicit committed BC prefix, never a trainer."""
    from gym_trading_env.research.fg_s3_e_quality_online_checks import calibration,require_calibration
    from gym_trading_env.research.fg_replay import write_json,signed,digest
    from .fg_s3_e_quality_online_probe import isolated_rl_probe,state_digest
    from .fg_checkpoint import host
    state=run.rl_lifecycle.state;policy=run.rl_lifecycle.policy
    run._assert_frozen_wm()
    run.agent.set_avail_actor_enabled(False)
    run.rl_lifecycle.transition('WM_CALIBRATING')
    state['attempt']+=1;state['technical']=None
    run.gates.reset_technical()
    attempt=run.root/'rl_admission'/f'attempt_{state["attempt"]:012d}'
    while attempt.exists():
        state['attempt']+=1;attempt=run.root/'rl_admission'/f'attempt_{state["attempt"]:012d}'
    attempt.mkdir(parents=True,exist_ok=False)
    run._save_checkpoint()  # Recoverable anchor; no candidate state can enter it.
    try:
        if not run.gates.ready['SUPERVISED_STABLE']:
            raise ValueError('ONLINE_SUPERVISED_SUPPORT_OR_STABILITY_INSUFFICIENT')
        run.verify_collection()
        history=fork_history(run.source.history,seed=policy['calibration_seed'])
        records=[];candidate_batch=None;calibration_files=[]
        try:
            with run.stream.pause(),run.collector.commit_lock:
                before=capture_live(run,attempt/'calibration_live_before')
                try:
                    for number in range(policy['calibration_batches']):
                        data,audit=history.sample()
                        natural=[i for i,row in enumerate(audit) if row['role']=='NATURAL']
                        causal={k:v[natural] for k,v in data.items()}
                        owners=[fingerprint(dict(corpus_id=audit[i]['corpus_id'],
                            source_identity=audit[i]['source_identity'],account_owner=audit[i]['rows'][-1]['owner'])) for i in natural]
                        record=calibration(run.agent,causal,source_identity=fingerprint(dict(
                            source_mix=history.identity,role='NATURAL',scope='CERTIFIED_TRAIN_WINDOWS_NOT_MARKET_FREQUENCY')),
                            horizon=policy['calibration_horizon'],seed=policy['calibration_seed']+number,source_step_owners=owners)
                        report=require_calibration(record)
                        path=attempt/f'calibration_{number:06d}.json'
                        write_json(path,signed(dict(calibration=report,source_windows=[audit[i] for i in natural],
                            input_batch_sha256=state_digest(causal))));records.append(record)
                        calibration_files.append(dict(path=str(path.relative_to(run.root)),sha256=digest(path)))
                        candidate_batch=data
                finally:
                    after=capture_live(run,attempt/'calibration_live_after')
                    write_json(attempt/'calibration_live_equivalence.json',signed(dict(
                        scope='NATIVE_READ_ONLY_CALIBRATION',equal=before==after,before=before,after=after)))
                    if before!=after:raise LiveStateChanged('ONLINE_NATIVE_CALIBRATION_CHANGED_LIVE_STATE')
        finally:history.close()
        for record in records:run.gates.calibration(record)
        if not run.gates.ready['WM_VALID_FOR_RL']:
            raise ValueError('ONLINE_WM_CALIBRATION_OR_UNIQUE_SUPPORT_INSUFFICIENT')
        run.rl_lifecycle.transition('RL_PROBING')
        run._save_checkpoint()
        context=dict(run_identity=run.account_verification.identity,market_step=int(run.step),
            source_bindings=run.source_bindings,collection=run.gates.evidence['collection'],
            calibration_files=calibration_files,wm_identity=run.gates.wm_identity,
            mask_identity=fingerprint(run.gates.policy))
        with run.stream.pause(),run.collector.commit_lock:
            before=capture_live(run,attempt/'numeric_live_before')
            try:
                probe=isolated_rl_probe(host(run.agent.save_exact()),run.settings,candidate_batch,
                    directory=attempt/'numeric',seed=policy['calibration_seed'],
                    controlled_small=run.identity['controlled_small'],context=context)
            finally:
                after=capture_live(run,attempt/'numeric_live_after')
                write_json(attempt/'numeric_live_equivalence.json',signed(dict(
                    scope='ISOLATED_RL_PROBE',equal=before==after,before=before,after=after)))
                if before!=after:raise LiveStateChanged('ONLINE_ISOLATED_PROBE_CHANGED_LIVE_STATE')
        if not run.gates.numeric(probe):raise ValueError('ONLINE_RL_NUMERIC_PROBE_NOT_READY')
        report=attempt/'numeric/candidate_report.json'
        technical=dict(context=context,calibration_files=calibration_files,
            numeric_file=dict(path=str(report.relative_to(run.root)),sha256=digest(report)),
            numeric_evidence=copy.deepcopy(run.gates.evidence['numeric']),
            source_agent_sha256=probe.report['source_agent_sha256'],
            calibration_state_unchanged=True,numeric_live_before=before,numeric_live_after=after,
            anchor_sequence=int(run.transaction)+1,
            gate_identity=run.gates.identity,wm_identity=run.gates.wm_identity,
            mask_identity=fingerprint(run.gates.policy))
        technical['evidence_hash']=fingerprint(technical)
        state['technical']=technical
        run.rl_lifecycle.transition('AWAIT_OPERATOR_REVIEW')
        run._save_checkpoint()
    except LiveStateChanged:
        raise  # Original committed anchor remains the only safe recovery point.
    except Exception as error:
        run.rl_lifecycle.transition('RL_BLOCKED',reason=str(error))
        run.agent.set_avail_actor_enabled(False)
        write_json(attempt/'blocked.json',signed(dict(status='RL_BLOCKED',reason=str(error),
            live_rl_updates=run.counters.state['rl_updates'],formal_rl_admitted=False)))
        run._save_checkpoint()


def validate_technical(run):
    from gym_trading_env.research.fg_replay import digest,read_manifest
    from gym_trading_env.research.fg_s3_e_quality_course_preflight import _local_path
    from gym_trading_env.research.fg_s3_e_quality_online_evidence import validate_account_report
    technical=run.rl_lifecycle.state['technical']
    if (not isinstance(technical,dict) or technical.get('evidence_hash')!=fingerprint(
            {k:v for k,v in technical.items() if k!='evidence_hash'})
            or technical.get('numeric_evidence')!=run.gates.evidence.get('numeric')
            or technical.get('gate_identity')!=run.gates.identity
            or technical.get('wm_identity')!=run.gates.wm_identity
            or technical.get('mask_identity')!=fingerprint(run.gates.policy)
            or technical['context']['source_bindings']!=run.source_bindings
            or technical['numeric_live_before']!=technical['numeric_live_after']):
        raise ValueError('ONLINE_RL_TECHNICAL_BINDING_CHANGED')
    for file in [*technical['calibration_files'],technical['numeric_file']]:
        if Path(file['path']).is_absolute():raise ValueError('ONLINE_RL_EVIDENCE_PATH_INVALID')
        path=_local_path(run.root/file['path'])
        if not path.is_relative_to(run.root) or digest(path)!=file['sha256']:
            raise ValueError('ONLINE_RL_TECHNICAL_EVIDENCE_CHANGED')
    validate_account_report(technical['context']['collection'],run.root,run.account_verification.identity)
    return technical


def review_contract(run):
    from .fg_checkpoint import validate_bundle
    from gym_trading_env.research.fg_replay import digest
    technical=validate_technical(run)
    node=run.root/'ckpt'/f'step_{technical["anchor_sequence"]:012d}'
    validate_bundle(node,run.identity)
    return dict(protocol='FG_S3_E_QUALITY_RL_REVIEW_BINDINGS_V2',
        run_identity=run.account_verification.identity,gate_identity=run.gates.identity,
        f3_anchor=dict(manifest=run.identity['source_manifest'],lineage=run.identity['source_run_identity'],
            source_nonopt=run.identity['source_nonopt']),sources=run.source_bindings,
        wm_identity=run.gates.wm_identity,mask_identity=fingerprint(run.gates.policy),wm_policy=run.gates.wm_policy,
        technical_evidence_hash=technical['evidence_hash'],collection=technical['context']['collection'],
        anchor_checkpoint=dict(node=node.name,manifest_sha256=digest(node/'manifest.json')),
        scope=dict(warmup_env_steps=run.rl_lifecycle.policy['warmup_env_steps'],
            env_steps=run.spec['env_steps'],max_optimizer_updates=run.settings['optimizer_updates'],
            max_consumed_batches=run.spec['max_consumed_batches'],
            continuation='NATIVE_ENV_CAUSAL_READER_GUARDED_FINAL_FULL_RECONCILE_REQUIRED'))
