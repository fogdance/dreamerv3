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
        self.state.update(cycles=0,warmup_cycles=0,rl_updates=0)


class OnlineRun(ActorRun,LiveRun):
    """Existing HeadsRun.update + LiveRun.collect_to, one Agent/optimizer/CP."""
    protocol='FG_S3_E_QUALITY_ONLINE_RUN_V1'

    def __init__(self,spec,factory,source,history=None,*,resume=False,authorized=False,
                 native_preparation=None,_probe=None,_test_location=None,wm_policy=None):
        from dreamerv3.fg_s3_e_quality_online import validate_execution,execution_contract
        from dreamerv3.fg_s3a_actor import parameter_digests
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
        if permit is not None and not engineering:raise ValueError('ONLINE_ENGINEERING_PERMIT_IDENTITY_MISMATCH')
        if not engineering and self.mode!='SHADOW':require_training_authority(native_preparation,authorized=authorized)
        if _test_location is not None and not engineering:raise ValueError('ONLINE_TEST_RELOCATION_FORBIDDEN')
        if source.get('fold')!='TRAIN':raise ValueError('ONLINE_TRAIN_SOURCE_REQUIRED')
        if self.mode!='SHADOW' and history is None:raise ValueError('ONLINE_HISTORICAL_READER_REQUIRED')
        if history is not None:history.validate_snapshot()
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
        chunksize=32 if engineering else 1024
        self.identity=dict(protocol=self.protocol,spec=s,execution_contract=execution_contract(self.mode),
            source_manifest=proof['manifest_sha256'],source_run_identity=proof['run_identity_sha256'],
            course=proof['identity']['preparation']['course'],source_nonopt=fingerprint({k:v for k,v in parameter_digests(state['params']).items() if not k.startswith('opt/')}),
            history=None if history is None else history.identity_hash,live_source=copy.deepcopy(source),
            env_config_hash=env.config_hash,e_identity=env.e_observation_identity,
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
        self.gates=OnlineGates(fingerprint(self.identity),formal_mask_policy(self.settings),wm_policy=wm_policy)
        self.native_preparation=native_preparation;self.authorized=authorized
        self.factory=factory
        chunk_identity=ChunkIdentity(fingerprint(self.identity))
        self.collector=OnlineCollector(env,source,self.root/'rollout',Path(__file__).resolve().parents[3]/'dreamerv3',
            agent=self.agent,seed=s['env_seed'],budget=s['env_steps'],capacity=s['replay_capacity'],existing=resume,
            replay_length=1+self.settings['burnin_steps']+self.settings['target_steps'],chunksize=chunksize,
            exploration_rate=s['exploration_rate'],exploration_seed=s['replay_seed'],chunk_identity=chunk_identity,
            execution_identity=fingerprint(self.identity))
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
        self.cp.runtime=self.runtime;self.cp.logs=JournalState(self.root,('updates.jsonl','cycles.jsonl'))
        self.cp.phase=self.phase
        self.cp.source_exposures=self.source_exposures
        if resume:
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

    def _save_checkpoint(self):
        pause=self.stream.pause() if self.stream is not None else contextlib.nullcontext()
        with pause,self.collector.commit_lock:
            self.transaction.increment();self.cp.save()

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
        if rl:
            self.counters.state['rl_updates']+=int(updated)
            self.gates.numeric(metrics,actor_changed=any(before[k]!=after[k] for k in before if k.startswith('quality_actor/')),
                value_changed=any(before[k]!=after[k] for k in before if k.startswith('val/')),frozen_equal=frozen)
        audit=self.stream.state.get('last_batch_audit',[])
        self.source_exposures.add(audit,self.last_batch,1+self.settings['burnin_steps'])
        for row in audit:
            if row.get('corpus_id')=='ONLINE_LIVE':
                for r in row['rows'][1+self.settings['burnin_steps']:]:
                    sid=r['stepid'];self.publisher.consumed[sid]=self.publisher.consumed.get(sid,0)+1

    def update(self):
        from dreamerv3.fg_s3a_actor import parameter_digests
        self._before_update=parameter_digests(self.agent.params)
        return HeadsRun.update(self)

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

    def verify_collection(self):
        from gym_trading_env.research.fg_replay import reconcile_trace,E_QUALITY_SCHEMA
        from gym_trading_env.research.fg_single3_evaluation import embodied_runtime
        out=self.root/'collection_checks'/f'step_{int(self.step):012d}'
        self.collector.snapshot_trace(out,dict(protocol=self.protocol,scope='NATIVE_ACCOUNT_INTEGRITY_ONLY'))
        proof=reconcile_trace(out,Path(__file__).resolve().parents[3]/'dreamerv3',
            env_factory=self.factory,schema=E_QUALITY_SCHEMA,runtime_factory=embodied_runtime)
        if not self.gates.collection(self.collector,proof):raise ValueError('ONLINE_COLLECTION_GATE_CLOSED')
        return proof

    def run_to(self,target=None,*,_stop_after_publications=None):
        s=self.spec;target=s['env_steps'] if target is None else target
        if (type(target) is not int or not int(self.step)<=target<=s['env_steps']
                or target!=s['env_steps'] and target%s['collect_every']):
            raise ValueError('ONLINE_STOP_BOUNDARY_INVALID')
        if _stop_after_publications is not None and not self.identity['controlled_small']:
            raise ValueError('ONLINE_PUBLICATION_INTERRUPT_PROBE_FORBIDDEN')
        phase=self.phase.state
        while int(self.step)<target or phase['stage']!='IDLE':
            if phase['stage']=='IDLE':phase.update(stage='COLLECT',end=min(int(self.step)+s['collect_every'],target),updates_done=0)
            if phase['stage']=='COLLECT':
                self.collect_to(phase['end'])
                if not self.gates.ready['COLLECTION_VALID']:self.verify_collection()
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
                        else:
                            try:self.gates.require_rl(self.native_preparation,authorized=self.authorized)
                            except ValueError:self.agent.set_avail_actor_enabled(False)
                            else:self.agent.set_avail_actor_enabled(True)
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
        self.cp.save(final=target==s['env_steps'])
        return self.result()

    def result(self):
        return dict(protocol=self.protocol,status='ONLINE_ENGINEERING_COMPLETE' if int(self.step)==self.spec['env_steps'] and self.phase.state['stage']=='IDLE' else 'ONLINE_ENGINEERING_PARTIAL',
            execution_mode=self.mode,env_steps=int(self.step),counters=self.counters.save(),
            gate=self.gates.report(),phase=self.phase.save(),source_exposures=self.source_exposures.save(),checkpoint=str(self.cp.latest()),
            training_admitted=self.identity['training_admitted'],real_TRAIN_updates=0 if self.identity['controlled_small'] else int(self.agent.n_updates),
            formal_rl_admitted=self.gates.report()['formal_rl_admitted'],research_admitted=False)

    def close(self):
        if self.stream is not None:
            with self.stream.pause():
                if self.source is not None:self.source.close()
        if self.publisher is not None:self.publisher.close()
        self.collector.close()
