"""Thin native online coordination; shared collection and update lifecycle."""
from __future__ import annotations

import contextlib
import copy
import json
import math
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import elements
import numpy as np

from .fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,host
from .fg_s3_e_quality_actor import ActorRun,ActorCounters
from .fg_s3_e_quality_heads import HeadsRun
from .fg_s3_v2_train import LiveRun
from .fg_s3_e_quality_online import inspect_f3
from .fg_s3_e_quality_online_stream import OnlineHistoryStream
from .fg_s3_e_quality_online_state import ChunkIdentity,JournalState,OnlinePhase,SourceExposures
from .fg_s3_e_quality_online_gate import OnlineGates

_PERMITS={}


def engineering_permit(spec,*,numeric_rl=False):
    """Fixed original synthetic identity, never editable real TRAIN admission."""
    from .fg_s3_e_quality_foundation import synthetic_probe_permit
    from gym_trading_env.research.fg_s3_e_quality_actor import preflight
    from gym_trading_env.envs.entry_contract import fingerprint
    proof,_=inspect_f3(spec,allow_engineering=True)
    if not proof['engineering_only']:raise ValueError('ONLINE_ENGINEERING_SOURCE_REQUIRED')
    p=preflight(proof['identity']['preparation']['settings']);synthetic_probe_permit(p)
    token=object();_PERMITS[token]=(fingerprint(spec),proof['manifest_sha256'],bool(numeric_rl))
    return token


class OnlineCounters(ActorCounters):
    def __init__(self):
        super().__init__()
        self.state.update(cycles=0,warmup_cycles=0,rl_updates=0,bc_updates=0)


class OnlineRun(ActorRun,LiveRun):
    """Existing HeadsRun.update + LiveRun.collect_to, one Agent/optimizer/CP."""
    protocol='FG_S3_E_QUALITY_ONLINE_RUN_V3'

    def __init__(self,spec,factory,source,history=None,*,resume=False,authorized=False,
                 native_preparation=None,_probe=None,_test_location=None,wm_policy=None,initial_state=None,
                 rl_review=None,source_bindings=None,native_audit=None,rl_policy=None):
        from dreamerv3.fg_s3_e_quality_online import validate_execution,execution_contract
        from dreamerv3.fg_s3a_actor import parameter_digests
        from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
        from gym_trading_env.envs.entry_contract import fingerprint
        from gym_trading_env.research.fg_replay import read_manifest,write_json,signed
        from gym_trading_env.research.fg_s3_e_quality_actor import create_agent,require_training_authority
        from gym_trading_env.research.fg_s3_e_quality_foundation import execution_identity
        from gym_trading_env.research.fg_s3_e_quality_online_collector import OnlineCollector
        from gym_trading_env.research.fg_s3_e_quality_online_data import LabelPublisher,OnlineWindowReader
        from gym_trading_env.research.fg_s3_e_quality_online_checks import formal_mask_policy
        from .fg_s3_e_quality_transfer import transfer_parameters
        from .fg_s3_source_admission import runtime_versions
        s=validate_execution(spec);self.spec=s;self.mode=s['execution_mode']
        permit=_PERMITS.get(_probe);proof,state=inspect_f3(s,allow_engineering=permit is not None)
        engineering=permit is not None and permit[:2]==(fingerprint(s),proof['manifest_sha256'])
        numeric_probe=bool(engineering and permit[2])
        from .fg_s3_e_quality_online_rl import validate_rl_policy,RLAdmissionState
        if not numeric_probe:rl_policy=validate_rl_policy(rl_policy,s)
        elif rl_policy is not None:raise ValueError('ONLINE_DEV_NUMERIC_PROBE_HAS_NO_FORMAL_RL_POLICY')
        if permit is not None and not engineering:raise ValueError('ONLINE_ENGINEERING_PERMIT_IDENTITY_MISMATCH')
        if not engineering and self.mode!='SHADOW':require_training_authority(native_preparation,authorized=authorized)
        if not engineering:
            from gym_trading_env.research.fg_s3_e_quality_online_admission import require_native_audit
            require_native_audit(native_audit,source_bindings)
        if _test_location is not None and not engineering:raise ValueError('ONLINE_TEST_RELOCATION_FORBIDDEN')
        if source.get('fold')!='TRAIN':raise ValueError('ONLINE_TRAIN_SOURCE_REQUIRED')
        if self.mode!='SHADOW' and history is None:raise ValueError('ONLINE_HISTORICAL_READER_REQUIRED')
        if history is not None:
            from gym_trading_env.research.fg_s3_e_quality_online import validate_history_contract
            history.validate_snapshot();validate_history_contract(history,proof)
        self.root=Path(_test_location or s['run_dir']).resolve()
        protected=[Path(s['source_run']),Path(proof['identity']['preparation']['settings']['corpus_root'])]
        if history is not None:protected.extend(Path(r.admission.source.corpus_root) for r in history.readers)
        for path in protected:
            if self.root.is_relative_to(path) or path.is_relative_to(self.root):raise ValueError('ONLINE_OUTPUT_OVERLAPS_SOURCE')
        if not resume and self.root.exists():raise ValueError('ONLINE_RUN_OUTPUT_MUST_BE_NEW')
        env=factory()
        if env.entry_config.get('state_path'):env.close();raise ValueError('ONLINE_PERSISTENT_OLD_ACCOUNT_FORBIDDEN')
        if engineering and (env._e_profile.identity!='c'*64 or env.contract!='CZCE.FG2701'
                            or source.get('scope')!='BUILTIN_SYNTHETIC_ONLY'):
            env.close();raise ValueError('ONLINE_BUILTIN_SYNTHETIC_ENV_REQUIRED')
        self.settings=copy.deepcopy(proof['identity']['preparation']['settings'])
        if rl_policy is not None and rl_policy['calibration_horizon']>=self.settings['target_steps']:
            env.close();raise ValueError('ONLINE_CALIBRATION_HORIZON_INVALID')
        self.settings['model_init']['seed']=s['model_seed'];self.settings['device']='cpu'
        geometry={k:self.settings[k] for k in ('batch_size','burnin_steps','target_steps')}
        if history is not None:
            geometry=copy.deepcopy(history.identity['settings'])
            if any(geometry[k]!=self.settings[k] for k in ('burnin_steps','target_steps')):
                env.close();raise ValueError('ONLINE_HISTORY_SOURCE_GEOMETRY_MISMATCH')
            self.settings['batch_size']=geometry['batch_size']*len(history.role_streams)
            self.settings['aux_sampling_spec']=None  # Effective Agent batch; Reader roles have their own frozen identity.
        cycles=math.ceil(s['env_steps']/s['collect_every'])
        self.settings.update(optimizer_updates=max(1,cycles*s['updates_per_collect']),max_consumed_batches=s['max_consumed_batches'])
        mask_policy=formal_mask_policy(self.settings)
        chunksize=32 if engineering else 1024
        self.identity=dict(protocol=self.protocol,spec=s,execution_contract=execution_contract(self.mode),
            source_manifest=proof['manifest_sha256'],source_run_identity=proof['run_identity_sha256'],
            frozen_wm_identity=fingerprint({k:v for k,v in parameter_digests(state['params']).items() if k.startswith(WM_GROUPS)}),
            course=proof['identity']['preparation']['course'],source_nonopt=fingerprint({k:v for k,v in parameter_digests(state['params']).items() if not k.startswith('opt/')}),
            history=None if history is None else history.identity_hash,live_source=copy.deepcopy(source),
            source_bindings=copy.deepcopy(source_bindings),rl_admission_policy=copy.deepcopy(rl_policy),
            online_settings=copy.deepcopy(self.settings),mask_policy=mask_policy,wm_policy=copy.deepcopy(wm_policy),
            env_config_hash=env.config_hash,e_identity=env.e_observation_identity,
            initial_account=None if initial_state is None else fingerprint(initial_state),
            chunksize=chunksize,code=execution_identity(),packages=runtime_versions(),
            controlled_small=engineering,numeric_probe=bool(engineering and permit[2]),training_admitted=not engineering and self.mode!='SHADOW',
            checkpoint_position='ATOMIC_TRANSACTION_SEQUENCE',checkpoint_policy='STRICT_LATEST_NO_FALLBACK')
        if resume and read_manifest(self.root/'run_identity.json')['identity']!=self.identity:
            env.close();raise ValueError('ONLINE_RESUME_IDENTITY_MISMATCH')
        if resume:
            selected=FullCheckpoint(self.root/'ckpt',elements.Counter(),self.identity,(),strict_latest=True).latest()
            if selected is None:env.close();raise ValueError('ONLINE_RESUME_CHECKPOINT_MISSING')
        self.agent=create_agent(self.settings,self.root/'model',controlled_small=engineering,execution_mode=self.mode)
        if not resume:
            transfer=transfer_parameters(self.agent,state,proof,target_stage='ACTOR_ONLINE')
            self.root.mkdir(parents=True,exist_ok=True)
            write_json(self.root/'run_identity.json',signed(dict(identity=self.identity)))
            write_json(self.root/'source_to_online_transfer.json',transfer)
        self.gates=OnlineGates(fingerprint(self.identity),formal_mask_policy(self.settings),
            wm_policy=wm_policy,collection_root=self.root)
        from gym_trading_env.research.fg_s3_e_quality_online_evidence import FinalAccountState
        self.account_verification=FinalAccountState(self.root,fingerprint(self.identity))
        self.rl_lifecycle=RLAdmissionState(fingerprint(self.identity),rl_policy,numeric_probe=numeric_probe)
        self.native_preparation=native_preparation;self.authorized=authorized
        self.source_bindings=copy.deepcopy(source_bindings)
        self.native_audit=copy.deepcopy(native_audit)
        self.rl_review=rl_review
        self.factory=factory
        chunk_identity=ChunkIdentity(fingerprint(self.identity))
        self.collector=OnlineCollector(env,source,self.root/'rollout',Path(__file__).resolve().parents[3]/'dreamerv3',
            agent=self.agent,seed=s['env_seed'],budget=s['env_steps'],capacity=s['replay_capacity'],existing=resume,
            replay_length=1+self.settings['burnin_steps']+self.settings['target_steps'],chunksize=chunksize,
            exploration_rate=s['exploration_rate'],exploration_seed=s['replay_seed'],chunk_identity=chunk_identity,
            execution_identity=fingerprint(self.identity),initial_state=initial_state)
        self.publisher=None;self.source=None;self.stream=None
        if self.mode!='SHADOW':
            self.publisher=LabelPublisher(self.collector,factory,source)
            online=OnlineWindowReader(self.collector,self.publisher,geometry,s['replay_seed'])
            self.source=OnlineHistoryStream(history,online,history_ratio=s['history_ratio'],seed=s['replay_seed'])
            self.stream=self.agent.stream(self.source)
        self.step=elements.Counter();self.transaction=elements.Counter();self.consumed_step=elements.Counter()
        self.counters=OnlineCounters();self.samples=[0];self.metrics=[];self.last_batch=None
        self.phase=OnlinePhase()
        self.source_exposures=SourceExposures()
        self.schedule=StepSchedule(s['collect_every'],s['save_every'])
        def set_carry(value):
            if value is not None:raise ValueError('ONLINE_RANDOM_WINDOW_CARRY_MUST_BE_FRESH')
        self.runtime=RuntimeState(self.schedule,SimpleNamespace(),[] if self.stream is None else [self.stream],
            lambda:None,set_carry,{},self.samples)
        self.cp=FullCheckpoint(self.root/'ckpt',self.transaction,self.identity,[] if self.stream is None else [self.stream],
            keep_all=True,reserve_gib=1,strict_latest=True)
        self.cp.replay=self.collector.replay;self.cp.driver=self.collector.driver;self.cp.collector=self.collector
        if self.publisher is not None:self.cp.publisher=self.publisher
        if self.source is not None:self.cp.course_reader=self.source
        self.cp.agent=self.agent;self.cp.step=self.transaction;self.cp.market_step=self.step
        self.cp.consumed_step=self.consumed_step;self.cp.counters=self.counters;self.cp.gates=self.gates
        self.cp.runtime=self.runtime;self.cp.logs=JournalState(self.root,('updates.jsonl','cycles.jsonl','rl_authorizations.jsonl'))
        self.cp.phase=self.phase
        self.cp.source_exposures=self.source_exposures
        self.cp.account_verification=self.account_verification
        self.cp.rl_lifecycle=self.rl_lifecycle
        if resume:
            self._validate_account_checkpoint(selected)
            self._validate_rl_checkpoint(selected)
            self.cp.load()
            if (int(self.step)!=self.collector.counts['env_steps'] or int(self.consumed_step)!=self.counters.state['consumed_batches']
                    or int(self.agent.n_updates)!=self.counters.state['optimizer_updates']):
                raise ValueError('ONLINE_RESTORED_COUNTER_MISMATCH')
        else:self._save_checkpoint()

    def _consume_batch_boundary(self):
        self.consumed_step.increment();self.phase.state['updates_done']+=1
    def _schedule_update(self,updated):return False,False
    def _check_consumption_budget(self):
        if self.counters.state['numerical_rejections']:raise ValueError('ONLINE_NUMERICAL_REJECTION_REQUIRES_REVIEW')
        if self.counters.state['consumed_batches']>=self.spec['max_consumed_batches']:
            raise ValueError('ONLINE_EXPLICIT_CONSUMPTION_BUDGET_EXHAUSTED')

    def _save_checkpoint(self,*,final=False):
        pause=self.stream.pause() if self.stream is not None else contextlib.nullcontext()
        with pause,self.collector.commit_lock:
            self.transaction.increment();self.cp.save(final=final)

    def _validate_account_checkpoint(self,node):
        """Cross-member prefix checks before restoring Agent or optimizer state."""
        from gym_trading_env.research.fg_s3_e_quality_online_evidence import prefix_rows_digest
        load=lambda name:pickle.loads((node/(name+'.pkl')).read_bytes())
        gates=load('gates');collector=load('collector');market=int(load('market_step'))
        account=load('account_verification')
        self.gates.validate_snapshot(gates);self.account_verification.validate_snapshot(account)
        report=gates['evidence'].get('collection')
        if report is not None:
            r=report['receipt'];prefix=r['prefix'];through=r['verified_through_market_step']
            if (through>market or prefix['rows']>len(collector['rows'])
                    or prefix_rows_digest(collector['rows'][:prefix['rows']])!=prefix['rows_sha256']
                    or gates['ready']['COLLECTION_VALID'] and through!=market):
                raise ValueError('ONLINE_CHECKPOINT_ACCOUNT_PREFIX_MISMATCH')
        if account['status']=='VERIFIED':
            final=account['final_evidence']
            if (report!=final or market!=self.spec['env_steps']
                    or final['receipt']['verified_through_market_step']!=market):
                raise ValueError('ONLINE_CHECKPOINT_FINAL_ACCOUNT_RANGE_MISMATCH')

    def _validate_rl_checkpoint(self,node):
        """Validate saved technical identities and actual RL ledger before Agent load."""
        import math
        from dreamerv3.fg_s3a_actor import parameter_digests
        from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
        from gym_trading_env.envs.entry_contract import fingerprint
        from .fg_s3_e_quality_online_rl import validate_frozen_online_state
        from .fg_checkpoint import validate_bundle
        load=lambda name:pickle.loads((node/(name+'.pkl')).read_bytes())
        lifecycle=load('rl_lifecycle');self.rl_lifecycle.validate_snapshot(lifecycle)
        gates=load('gates');counters=load('counters');agent=load('agent');logs=load('logs')
        self.cp._saveables['logs'].validate_snapshot(logs)
        size=logs['updates.jsonl']['bytes']
        rows=[json.loads(line) for line in (self.root/'updates.jsonl').read_bytes()[:size].splitlines()] if size else []
        updates=rl=bc=0
        for number,row in enumerate(rows,1):
            metrics=row['metrics'];updates+=int(bool(metrics['f3/updated']))
            rl+=int(bool(metrics.get('f4/rl_active')) and bool(metrics['f3/updated']))
            bc+=int(bool(metrics.get('f3/active')) and bool(metrics['f3/updated']))
            if (row['consumed_batches']!=number or row['local_update']!=updates
                    or row['agent_update']!=updates or metrics['opt/updates']!=updates
                    or not metrics['f3/finite'] or not math.isfinite(metrics['opt/loss'])
                    or not math.isfinite(metrics['opt/grad_norm'])):
                raise ValueError('ONLINE_CHECKPOINT_UPDATE_LEDGER_INVALID')
        if (len(rows)!=counters['consumed_batches'] or updates!=counters['optimizer_updates']
                or updates!=agent['counters']['updates'] or rl!=counters['rl_updates'] or bc!=counters['bc_updates']):
            raise ValueError('ONLINE_CHECKPOINT_RL_OR_BC_COUNTER_MISMATCH')
        initial=self.root/'ckpt/step_000000000001'
        validate_bundle(initial,self.identity)
        first=pickle.loads((initial/'agent.pkl').read_bytes())
        groups=() if self.mode=='SHADOW' else ('quality_actor/',)
        if rl>0:groups+=('val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/','avail_actor_gate/')
        validate_frozen_online_state(first['params'],agent['params'],self.agent.model.config,
            groups,updated=updates>0)
        if lifecycle['stage'] in ('AWAIT_OPERATOR_REVIEW','RL_ACTIVE','FINAL_VERIFY','COMPLETE_VERIFIED'):
            technical=lifecycle['technical']
            if (technical['evidence_hash']!=fingerprint({k:v for k,v in technical.items() if k!='evidence_hash'})
                    or technical['numeric_evidence']!=gates['evidence']['numeric']
                    or fingerprint({k:v for k,v in parameter_digests(agent['params']).items()
                        if k.startswith(WM_GROUPS)})!=gates['wm_identity']):
                raise ValueError('ONLINE_CHECKPOINT_RL_TECHNICAL_OR_WM_CHANGED')
            if lifecycle['stage']=='COMPLETE_VERIFIED' and rl<=0:
                raise ValueError('ONLINE_RL_COMPLETION_REQUIRES_ACTUAL_UPDATES')

    def _accumulate_metrics(self,metrics,updated):
        from dreamerv3.fg_s3a_actor import parameter_digests
        super()._accumulate_metrics(metrics,updated)
        after=parameter_digests(self.agent.params);before=self._before_update
        rl=bool(metrics.get('f4/rl_active',False))
        allowed=('quality_actor/','val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/') if rl else ('quality_actor/',)
        def permitted(key):
            if key.startswith(allowed):return True
            owner=self.agent.model.f3_state_owner[key]
            return owner=='CLOCK' or owner=='quality_actor/' or (rl and 'val/' in key and key.startswith('opt/'))
        frozen=all(before[k]==after[k] for k in before if not permitted(k))
        if not frozen:raise RuntimeError('ONLINE_FROZEN_PARAMETER_OR_SLOT_CHANGED')
        self.gates.supervised(metrics,frozen_equal=frozen)
        self.counters.state['bc_updates']+=int(updated and bool(metrics.get('f3/active',False)))
        if rl:
            self.counters.state['rl_updates']+=int(updated)
            self.gates.observe_rl(metrics,frozen_equal=frozen)
        audit=self.stream.state.get('last_batch_audit',[])
        self.source_exposures.add(audit,self.last_batch,1+self.settings['burnin_steps'])
        for row in audit:
            if row.get('corpus_id')=='ONLINE_LIVE':
                for r in row['rows'][1+self.settings['burnin_steps']:]:
                    sid=r['stepid'];self.publisher.consumed[sid]=self.publisher.consumed.get(sid,0)+1

    def update(self):
        from dreamerv3.fg_s3a_actor import parameter_digests
        self._before_update=parameter_digests(self.agent.params)
        if self.mode=='ONLINE_RL_CONTROLLED' and not self.identity['numeric_probe']:
            if int(self.step)>self.rl_lifecycle.policy['warmup_env_steps']:
                self._require_live_rl(self._before_update)
            elif self.agent.get_avail_actor_enabled():
                raise ValueError('ONLINE_RL_FORBIDDEN_DURING_BC_WARMUP')
        return HeadsRun.update(self)

    def _require_live_rl(self,digests=None):
        from dreamerv3.fg_s3a_actor import parameter_digests
        from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
        from gym_trading_env.envs.entry_contract import fingerprint
        if self.rl_lifecycle.state['stage']!='RL_ACTIVE':raise ValueError('ONLINE_RL_LIFECYCLE_NOT_AUTHORIZED')
        digests=parameter_digests(self.agent.params) if digests is None else digests
        if (fingerprint({k:v for k,v in digests.items() if k.startswith(WM_GROUPS)})!=self.gates.wm_identity
                or fingerprint({k:v for k,v in digests.items() if k.startswith(WM_GROUPS)})!=self.identity['frozen_wm_identity']
                or self.source_bindings!=self.identity['source_bindings']
                or self.source.history.identity_hash!=self.identity['history']
                or int(self.step)>self.spec['env_steps']
                or int(self.agent.n_updates)>self.settings['optimizer_updates']
                or self.counters.state['consumed_batches']>self.spec['max_consumed_batches']):
            self.gates.formal_admitted=False
            raise ValueError('ONLINE_RL_WM_IDENTITY_OR_REVIEW_SCOPE_CHANGED')
        self.gates.require_rl(self.native_preparation,authorized=self.authorized,review=self.rl_review)

    def _assert_frozen_wm(self):
        from dreamerv3.fg_s3a_actor import parameter_digests
        from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
        from gym_trading_env.envs.entry_contract import fingerprint
        actual=fingerprint({k:v for k,v in parameter_digests(self.agent.params).items() if k.startswith(WM_GROUPS)})
        if actual!=self.identity['frozen_wm_identity']:
            raise RuntimeError('ONLINE_FROZEN_WM_REPLACED_REQUIRES_ORIGINAL_SAFE_CHECKPOINT')

    def reauthorize_rl(self,path,sha256,*,authorize_rl=False):
        from .fg_s3_e_quality_online_rl import review_contract
        from gym_trading_env.envs.entry_contract import fingerprint
        if self.mode!='ONLINE_RL_CONTROLLED' or self.identity['numeric_probe']:
            raise ValueError('ONLINE_REVIEW_REQUIRES_FORMAL_CONTROLLED_RL_LIFECYCLE')
        if self.rl_lifecycle.state['stage'] not in ('AWAIT_OPERATOR_REVIEW','RL_ACTIVE'):
            raise ValueError('ONLINE_REVIEW_REQUIRES_COMPLETED_TECHNICAL_EVIDENCE')
        contract=review_contract(self)
        self.rl_review=self.gates.review_rl(self.native_preparation,path,sha256,
            authorized=authorize_rl,expected_contract=contract)
        if self.rl_lifecycle.state['stage']=='AWAIT_OPERATOR_REVIEW':self.rl_lifecycle.transition('RL_ACTIVE')
        self.rl_lifecycle.state['last_review_sha256']=sha256
        self._require_live_rl()
        event=dict(protocol='FG_S3_E_QUALITY_RL_AUTHORIZATION_EVENT_V1',
            reviewed_sha256=sha256,review_contract_hash=fingerprint(contract),
            authorized_at_market_step=int(self.step),authorized_at_consumed_batches=self.counters.state['consumed_batches'],
            optimizer_updates=int(self.agent.n_updates),cold_process_revalidated=True)
        with (self.root/'rl_authorizations.jsonl').open('a') as stream:
            stream.write(json.dumps(event,sort_keys=True)+'\n')
        self._save_checkpoint()

    def _rl_boundary(self):
        if self.mode!='ONLINE_RL_CONTROLLED' or self.identity['numeric_probe']:return False
        if int(self.step)<self.rl_lifecycle.policy['warmup_env_steps']:return False
        stage=self.rl_lifecycle.state['stage']
        if stage in ('WARMUP','WM_CALIBRATING','RL_PROBING','RL_BLOCKED'):
            from .fg_s3_e_quality_online_rl import perform_admission
            perform_admission(self)
            return True
        return stage=='AWAIT_OPERATOR_REVIEW' or stage=='RL_ACTIVE' and self.rl_review is None

    def _reject_batch(self,error):
        self.consumed_step.increment();self.phase.state['updates_done']+=1;c=self.counters.state
        c['consumed_batches']+=1;c['numerical_rejections']+=1
        with (self.root/'updates.jsonl').open('a') as f:
            f.write(json.dumps(dict(consumed_batches=c['consumed_batches'],rejection=str(error),optimizer_updates=int(self.agent.n_updates)))+'\n')
        self._save_checkpoint()

    def publish_labels(self,limit=None,*,save=True):
        if self.publisher is None:return 0
        pause=self.stream.pause() if self.stream is not None else contextlib.nullcontext()
        with pause:
            count=self.publisher.publish(limit)
            if save:self._save_checkpoint()
            return count

    def collect_to(self,end):
        super().collect_to(end)
        self.gates.collection_advanced(int(self.step))

    def verify_collection(self,*,final=False):
        from gym_trading_env.envs.entry_contract import fingerprint
        from gym_trading_env.research.fg_s3_e_quality_online_evidence import verify_account
        out=self.account_verification.allocate(int(self.step),final)
        evidence=verify_account(self.collector,self.factory,out,run_identity=fingerprint(self.identity),
            source_identity=fingerprint(self.identity['live_source']),verifier_identity=self.identity['code'],
            market_step=int(self.step),final=final)
        if not self.gates.collection(self.collector,evidence):raise ValueError('ONLINE_COLLECTION_GATE_CLOSED')
        return copy.deepcopy(evidence.report['receipt']['reconciliation'])

    def _final_account_commit(self):
        state=self.account_verification.state
        if state['status']=='VERIFIED':
            self.account_verification.validate_snapshot()
            return
        state.update(status='VERIFYING',error=None)
        try:
            self._validate_final_publication()
            self.verify_collection(final=True)
            state.update(status='VERIFIED',final_evidence=copy.deepcopy(self.gates.evidence['collection']))
            if self.rl_lifecycle.state['stage']=='FINAL_VERIFY':
                self.rl_lifecycle.transition('COMPLETE_VERIFIED')
            # A new transaction is essential: FullCheckpoint reuses an existing
            # node at the same sequence number without reserializing its members.
            self._save_checkpoint(final=True)
        except Exception as error:
            state.update(status='NO_GO',final_evidence=None,error=str(error))
            self.gates.ready['COLLECTION_VALID']=False
            if self.rl_lifecycle.state['stage']=='FINAL_VERIFY':
                self.rl_lifecycle.transition('RL_BLOCKED',reason=str(error))
            self._save_checkpoint(final=False)
            raise
        self._publish_completion()

    def _validate_final_publication(self):
        if self.publisher is None:return  # SHADOW has no teacher publication.
        if self.publisher.strict is None or self.publisher.blockers:
            raise ValueError('ONLINE_FINAL_SUPERVISION_UNAVAILABLE')
        for row in self.collector.rows[:-1]:
            entry=self.publisher.ledger.get(row['stepid'])
            if (entry is None or entry['status'] not in ('CERTIFIED','INVALID')
                    or type(entry['publication_seq']) is not int or entry['publication_seq']<=0):
                raise ValueError('ONLINE_FINAL_LEGAL_SIDECAR_NOT_PUBLISHED')
        # The final unexecuted decision stays PENDING/invalid by the existing
        # publication contract. It is not silently labeled WAIT.
        self.publisher.validate_snapshot()

    def _publish_completion(self):
        from gym_trading_env.research.fg_replay import digest,signed,read_manifest
        node=self.cp.latest()
        receipt=signed(dict(protocol='FG_S3_E_QUALITY_ONLINE_COMPLETION_V1',
            run_identity=self.account_verification.identity,transaction=int(self.transaction),
            checkpoint_manifest_sha256=digest(node/'manifest.json'),
            result=self.result(),full_account_evidence=self.account_verification.state['final_evidence']))
        path=self.root/'completion'/f'transaction_{int(self.transaction):012d}.json'
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            if read_manifest(path)!=receipt:raise ValueError('ONLINE_COMPLETION_RECEIPT_CHANGED')
            return path
        with path.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
        return path

    def publish_result(self,result):
        """Every invocation retains its own receipt; never overwrite old results."""
        import uuid
        from gym_trading_env.research.fg_replay import signed
        path=self.root/'results'/f'transaction_{int(self.transaction):012d}_{uuid.uuid4().hex}.json'
        path.parent.mkdir(exist_ok=True)
        with path.open('x') as stream:json.dump(signed(result),stream,sort_keys=True,indent=2);stream.write('\n')
        return path

    def run_to(self,target=None,*,_stop_after_publications=None):
        s=self.spec;target=s['env_steps'] if target is None else target
        if (type(target) is not int or not int(self.step)<=target<=s['env_steps']
                or target!=s['env_steps'] and target%s['collect_every']):
            raise ValueError('ONLINE_STOP_BOUNDARY_INVALID')
        if _stop_after_publications is not None and not self.identity['controlled_small']:
            raise ValueError('ONLINE_PUBLICATION_INTERRUPT_PROBE_FORBIDDEN')
        phase=self.phase.state
        if phase['stage']=='IDLE' and self._rl_boundary():return self.result()
        while int(self.step)<target or phase['stage']!='IDLE':
            if phase['stage']=='IDLE':phase.update(stage='COLLECT',end=min(int(self.step)+s['collect_every'],target),updates_done=0)
            if phase['stage']=='COLLECT':
                self.collect_to(phase['end'])
                if not self.gates.collection_through():self.verify_collection()
                phase['stage']='PUBLISH'
            if phase['stage']=='PUBLISH':
                if self.publisher is not None:
                    self.publish_labels(_stop_after_publications,save=False)
                    if _stop_after_publications is not None:
                        self._save_checkpoint();return self.result()
                phase['stage']='UPDATE'
            if phase['stage']=='UPDATE':
                if self.publisher is not None and self.source.online._eligible():
                    if self.mode=='ONLINE_RL_CONTROLLED':
                        if self.identity['numeric_probe']:self.agent.set_avail_actor_enabled(True)
                        elif int(self.step)<=self.rl_lifecycle.policy['warmup_env_steps']:
                            self.agent.set_avail_actor_enabled(False)
                        else:
                            try:self._require_live_rl()
                            except ValueError as error:
                                self.rl_lifecycle.transition('RL_BLOCKED',reason=str(error))
                                self._save_checkpoint();return self.result()
                            self.agent.set_avail_actor_enabled(True)
                    while phase['updates_done']<s['updates_per_collect']:self.update()
                    self.agent.sync_policy_from_train()
                elif self.publisher is not None:self.counters.state['warmup_cycles']+=1
                phase['stage']='FINALIZE'
            if phase['stage']=='FINALIZE':
                self.counters.state['cycles']+=1
                _,save=self.schedule.advance(self.step)
                with (self.root/'cycles.jsonl').open('a') as f:
                    f.write(json.dumps(dict(env_steps=int(self.step),optimizer_updates=int(self.agent.n_updates),
                        consumed_batches=self.counters.state['consumed_batches'],publication_seq=0 if self.publisher is None else self.publisher.sequence,
                        rl_updates=self.counters.state['rl_updates']),sort_keys=True)+'\n')
                phase.update(stage='IDLE',end=int(self.step),updates_done=0)
                if save or int(self.step)==target:self._save_checkpoint()
                if self._rl_boundary():return self.result()
        if target==s['env_steps']:
            if self.mode=='ONLINE_RL_CONTROLLED' and not self.counters.state['rl_updates']:
                self._save_checkpoint(final=False)
            else:
                if self.rl_lifecycle.state['stage']=='RL_ACTIVE':self.rl_lifecycle.transition('FINAL_VERIFY')
                self._final_account_commit()
                self._publish_completion()  # Also completes a commit-before-receipt crash.
        return self.result()

    def result(self):
        complete=int(self.step)==self.spec['env_steps'] and self.phase.state['stage']=='IDLE'
        verified=complete and self.account_verification.state['status']=='VERIFIED'
        lifecycle=self.rl_lifecycle.state['stage']
        if lifecycle=='RL_BLOCKED':status='RL_BLOCKED'
        elif lifecycle=='AWAIT_OPERATOR_REVIEW' or lifecycle=='RL_ACTIVE' and self.rl_review is None:
            status='AWAIT_OPERATOR_REVIEW'
        elif verified:
            status={'SHADOW':'SHADOW_COMPLETE_VERIFIED','ONLINE_SUPERVISED':'ONLINE_BC_COMPLETE_VERIFIED',
                'ONLINE_RL_CONTROLLED':'RL_COMPLETE_VERIFIED'}[self.mode]
        elif self.account_verification.state['status']=='NO_GO':status='COMPLETE_BUT_UNVERIFIED/NO_GO'
        elif complete and self.mode=='ONLINE_RL_CONTROLLED':status='AWAIT_OPERATOR_REVIEW'
        else:status='ONLINE_PARTIAL'
        return dict(protocol=self.protocol,status=status,
            execution_mode=self.mode,env_steps=int(self.step),counters=self.counters.save(),
            gate=self.gates.report(),phase=self.phase.save(),source_exposures=self.source_exposures.save(),checkpoint=str(self.cp.latest()),
            training_admitted=self.identity['training_admitted'],real_TRAIN_updates=0 if self.identity['controlled_small'] else int(self.agent.n_updates),
            formal_rl_admitted=self.gates.report()['formal_rl_admitted'],research_admitted=False,
            account_verification=self.account_verification.save(),rl_lifecycle=self.rl_lifecycle.save(),
            live_process_rl_authorized=self.gates.report()['formal_rl_admitted'])

    def close(self):
        if self.stream is not None:
            with self.stream.pause():
                if self.source is not None:self.source.close()
        if self.publisher is not None:self.publisher.close()
        self.collector.close()
