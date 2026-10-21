"""Thin F3 coordination over the original Reader, Agent and FullCheckpoint."""
from __future__ import annotations
import copy
import json,pickle
from pathlib import Path
import numpy as np
from .fg_s3_e_quality_foundation import CourseStream
from gym_trading_env.research.fg_s3_e_quality_actor import F3WindowAdapter
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader


class ActorStreams:
    """Compose native F3 role streams; sampling and RNG remain in WP03 Readers."""
    def __init__(self,preparation,index_path):
        self.streams=[];self._group_identity=None;self._failed=self._closed=False
        self.last_batch_audit=[];s=preparation.settings
        specs=[('NATURAL',s['sampling_spec'])]
        if s.get('aux_sampling_spec') is not None:specs.append(('AUXILIARY',s['aux_sampling_spec']))
        self.roles=[role for role,_ in specs]
        try:
            for i,(_,spec) in enumerate(specs):
                reader=EQualityCourseReader.open(preparation.source,stage='ACTOR_STRICT',sampling_spec=spec,
                    seed=s['model_init']['seed']+i,index_path=index_path)
                settings={**s,'sampling_spec':copy.deepcopy(spec)}
                self.streams.append(CourseStream(reader,settings,adapter_cls=F3WindowAdapter))
                self.streams[-1].adapter.validate_support()
        except BaseException:
            self.close();raise
        self._init_transactions()

    def _init_transactions(self):
        from gym_trading_env.research.fg_s3_reader_transactions import TransactionGroup
        self._transactions=TransactionGroup(self,self.streams,
            lambda:self.last_batch_audit,lambda state:setattr(self,'last_batch_audit',state))

    def begin_batch(self):return self._transactions.begin()
    def commit_batch(self,mark):self._transactions.commit(mark)
    def rollback_batch(self,mark):self._transactions.rollback(mark)

    @classmethod
    def from_streams(cls,role_streams,*,identity):
        """Compose ready lifecycle streams without adding a sampler or Run."""
        from gym_trading_env.envs.entry_contract import fingerprint
        roles=[role for role,_ in role_streams]
        if roles not in (['NATURAL'],['NATURAL','AUXILIARY']):
            raise ValueError('F3_READER_GROUP_ROLES_INVALID')
        result=cls.__new__(cls)
        result.roles=roles;result.streams=[stream for _,stream in role_streams]
        result._group_identity=fingerprint(identity)
        result._failed=result._closed=False;result.last_batch_audit=[]
        result._init_transactions()
        result.validate_snapshot()
        return result

    def _ensure_open(self):
        if self._closed or self._failed:raise ValueError('F3_READER_GROUP_INVALID_REOPEN_REQUIRED')

    @property
    def readers(self):return [reader for stream in self.streams for reader in stream.readers]
    def __iter__(self):return self
    def __next__(self):
        from gym_trading_env.research.fg_s3_reader_transactions import batch_transaction
        self._ensure_open()
        with batch_transaction(self):
            batches=[]
            for role,stream in zip(self.roles,self.streams):
                data=next(stream);data['f3_natural'][...]=role=='NATURAL';batches.append(data)
            result={k:np.concatenate([b[k] for b in batches],axis=0) for k in batches[0]}
            if self._group_identity is not None:
                self.last_batch_audit=[dict(role=role,**row) for role,stream in zip(self.roles,self.streams)
                                       for row in stream.last_batch_audit]
                if len(self.last_batch_audit)!=len(result['stepid']):raise ValueError('F3_READER_GROUP_AUDIT_SIZE')
            return result

    def save(self):
        return self._snapshot(full=True)
    def save_small(self):return self._snapshot(full=False)
    def _snapshot(self,*,full):
        self._ensure_open()
        state=dict(protocol='FG_S3_F3_READER_GROUP_V1',roles=list(self.roles),
                   readers=[s.save() if full else s.save_small() for s in self.streams])
        if self._group_identity is not None:
            state.update(protocol='FG_S3_F3_BOUND_READER_GROUP_V1',identity=self._group_identity,
                         last_batch_audit=copy.deepcopy(self.last_batch_audit))
        return state
    save_full=save
    def validate_snapshot(self,state=None):
        self._ensure_open()
        protocol='FG_S3_F3_BOUND_READER_GROUP_V1' if self._group_identity is not None else 'FG_S3_F3_READER_GROUP_V1'
        if state is not None and (state.get('protocol')!=protocol or state.get('roles')!=self.roles
                or len(state.get('readers',[]))!=len(self.streams)):raise ValueError('F3_READER_GROUP_IDENTITY_MISMATCH')
        if state is not None and self._group_identity is not None:
            expected=[dict(role=role,**row) for role,value in zip(self.roles,state['readers'])
                      for row in value.get('last_batch_audit',[])]
            if state.get('identity')!=self._group_identity or state.get('last_batch_audit')!=expected:
                raise ValueError('F3_READER_GROUP_IDENTITY_MISMATCH')
        for i,stream in enumerate(self.streams):stream.validate_snapshot(None if state is None else state['readers'][i])
    def _apply(self,state):
        for stream,value in zip(self.streams,state['readers']):stream.load(value)
        if self._group_identity is not None:self.last_batch_audit=copy.deepcopy(state['last_batch_audit'])
    def _rollback(self,state):
        for stream,value in zip(self.streams,state['readers']):
            try:stream.load(value)
            except BaseException:self._failed=True
        if self._group_identity is not None:self.last_batch_audit=copy.deepcopy(state['last_batch_audit'])
    def load(self,state):
        self.validate_snapshot(state)
        before=self.save()
        try:self._apply(state)
        except BaseException:
            self._rollback(before)
            raise
    load_full=load
    def coverage(self):return {role:stream.coverage() for role,stream in zip(self.roles,self.streams)}
    def close(self):
        if self._closed:return
        for stream in self.streams:stream.close()
        self._closed=True


_PERMITS={}
def engineering_permit(preparation):
    from .fg_s3_e_quality_foundation import synthetic_probe_permit
    synthetic_probe_permit(preparation)
    if not preparation.receipt['identity']['source_model']['engineering_only']:raise ValueError('F3_SYNTHETIC_SOURCE_REQUIRED')
    permit=object();_PERMITS[id(preparation)]=(preparation,permit)
    return permit


class ActorCounters:
    def __init__(self):
        self.state=dict(optimizer_updates=0,consumed_batches=0,skip_no_valid=0,skip_disabled_loss=0,numerical_rejections=0,
            target_positions=0,target_market_minutes=0,burnin_positions=0,burnin_market_minutes=0,
            exposures=dict.fromkeys(('natural_count','natural_add','natural_wait','auxiliary_count','auxiliary_add',
                'auxiliary_wait','balanced_count','strict_pair_count','quality_pair_count','quality_pair_candidates',
                'quality_pair_conflicts','quality_pair_same_action_filtered','quality_pair_reversed_filtered'),0))
    def save(self):return copy.deepcopy(self.state)
    def load(self,state):
        if (state.keys()!=self.state.keys() or state['exposures'].keys()!=self.state['exposures'].keys()
                or any(type(v) is not int or v<0 for k,v in state.items() if k!='exposures')
                or any(type(v) is not int or v<0 for v in state['exposures'].values())):raise ValueError('F3_COUNTER_SCHEMA_MISMATCH')
        if state['consumed_batches']!=sum(state[k] for k in ('optimizer_updates','skip_no_valid','skip_disabled_loss','numerical_rejections')):
            raise ValueError('F3_COUNTER_ACCOUNTING_MISMATCH')
        self.state=copy.deepcopy(state)


from .fg_s3_e_quality_heads import HeadsRun

class ActorRun(HeadsRun):
    """Stage hooks only. The original HeadsRun owns update/run/CP scheduling."""
    metrics_stage='f3'
    def __init__(self,preparation,index_path,root,*,resume=False,authorize_training=False,_probe=None):
        import elements
        from .fg_checkpoint import FullCheckpoint,plain
        from .fg_s3_e_quality_foundation import (BACKEND,RECOVERY_POLICY,validate_offline_journal,
            restored_run_identity,validate_restore_agent,restore_reader_binding)
        from .fg_s3_e_quality_transfer import inspect_stage,transfer_parameters
        from .fg_s3_source_admission import runtime_versions
        from gym_trading_env.research.fg_replay import read_manifest,write_json,signed,digest
        from gym_trading_env.research.fg_s3_e_quality_actor import preflight,require_training_authority,create_agent,execution_identity
        from gym_trading_env.research.fg_s3_reader_versions import (VersionedStream,validate_checkpoint_stream,
            STATE_PROTOCOL,PREFETCH_PROTOCOL)
        permit=_PERMITS.get(id(preparation))
        engineering=permit is not None and permit[0] is preparation and permit[1] is _probe
        if not engineering:require_training_authority(preparation,authorized=authorize_training)
        self.preparation=preparation;self.settings=s=preparation.settings
        self.root=Path(root).resolve();self.index=Path(index_path).resolve()
        for key in ('corpus_root','source_run'):
            source=Path(s[key]).resolve()
            if self.root.is_relative_to(source) or source.is_relative_to(self.root):raise ValueError('F3_RUN_OVERLAPS_SOURCE')
        selected=None
        if resume:
            self.identity=restored_run_identity(self.root,'FG_S3_E_QUALITY_F3_RUN_V1',
                preparation.source,self.index,engineering=engineering)
            selected=FullCheckpoint(self.root/'ckpt',elements.Counter(),self.identity,(),backend=BACKEND,strict_latest=True).latest()
            if selected is None:raise ValueError('F3_RESUME_CHECKPOINT_MISSING')
        else:
            if self.root.exists():raise ValueError('F3_RUN_OUTPUT_MUST_BE_NEW')
            self.identity=plain(dict(protocol='FG_S3_E_QUALITY_F3_RUN_V1',backend=BACKEND,recovery_policy=RECOVERY_POLICY,
                reader_stream_state_protocol=STATE_PROTOCOL,prefetch_pending_state_protocol=PREFETCH_PROTOCOL,
                preparation=preparation.receipt['identity'],
                compatibility=preparation.receipt['compatibility'],verifier_binding=preparation.receipt['verifier_binding'],
                index_sha256=digest(self.index),index_receipt_sha256=digest(self.index.with_suffix('.json')),
                code=execution_identity(),packages=runtime_versions(),controlled_small=engineering,training_admitted=not engineering,
                carry_rule='FRESH_PER_RANDOM_WINDOW',checkpoint_position='CONSUMED_BATCHES'))
        self.source=VersionedStream(ActorStreams(preparation,self.index))
        try:
            self.source.validate_snapshot()
            if selected is not None:restore_reader_binding(self.source,selected)
            if selected is not None:validate_checkpoint_stream(self.source,selected)
            if selected is not None:validate_offline_journal(self.root,selected,position_key='consumed_batches')
            self.agent=create_agent(s,self.root/'model',controlled_small=engineering)
            if resume:validate_restore_agent(self.agent,selected)
            if not resume:
                proof,state=inspect_stage(s,allow_engineering=engineering)
                receipt=transfer_parameters(self.agent,state,proof,target_stage='ACTOR_SUPERVISED')
                receipt.pop('content_hash');receipt.update(f3_run_identity=self.identity,
                    reader_seeds=[s['model_init']['seed']+i for i in range(len(self.source.source.streams))],
                    sampling_spec=s['sampling_spec'],aux_sampling_spec=s['aux_sampling_spec'])
                self.root.mkdir(parents=True)
                write_json(self.root/'run_identity.json',signed(dict(identity=self.identity)))
                write_json(self.root/'source_to_f3_transfer.json',signed(receipt))
            self._initialize_runtime(resume,engineering,counter_type=ActorCounters)
        except BaseException:
            self.source.close();raise

    def _accumulate_metrics(self,metrics,updated):
        for key in self.counters.state['exposures']:self.counters.state['exposures'][key]+=int(metrics['f3/'+key])

    def _reject_batch(self,error):
        super()._reject_batch(error)
        self.step.increment();c=self.counters.state;c['consumed_batches']+=1;c['numerical_rejections']+=1
        row=dict(local_update=c['optimizer_updates'],agent_update=int(self.agent.n_updates),consumed_batches=int(self.step),
            metrics={'f3/finite':False,'f3/updated':False,'f3/rejection':str(error),'opt/updates':int(self.agent.n_updates)})
        with (self.root/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.cp.save()

    def _check_consumption_budget(self):
        if self.counters.state['numerical_rejections']:raise ValueError('F3_NUMERICAL_REJECTION_REQUIRES_REVIEW')
        if self.counters.state['consumed_batches']>=self.settings['max_consumed_batches']:
            self.cp.save()
            raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: F3_CONSUMPTION_BUDGET_EXHAUSTED')

    def result(self):
        complete=int(self.agent.n_updates)==self.settings['optimizer_updates']
        return dict(status=('F3_ENGINEERING_COMPLETE' if complete else 'F3_ENGINEERING_PARTIAL') if self.identity['controlled_small'] else 'F3_TRAIN_AUTHORIZED',
            foundation_status=self.preparation.receipt['identity']['source_model']['foundation_status'],
            counters=self.counters.save(),training_admitted=self.identity['training_admitted'],model_research_go=False,
            checkpoint=str(self.cp.latest()),reader_coverage=self.source.coverage())
