"""Thin F2 stage coordinator over the original Agent/Reader/FullCheckpoint."""
from __future__ import annotations
import copy,json,pickle
from pathlib import Path
from types import SimpleNamespace
import elements
import numpy as np
from .fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,host,plain
from .fg_s3_e_quality_foundation import CourseStream,BACKEND
from .fg_s3_e_quality_transfer import inspect_f1,transfer_parameters
from .fg_s3_source_admission import runtime_versions
from .fg_s3c_train import reconcile_update_log
from gym_trading_env.research.fg_replay import read_manifest,write_json,signed,digest
from gym_trading_env.research.fg_s3_e_quality_heads import (
    preflight,require_training_authority,create_agent,execution_identity,F2WindowAdapter)
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader

_PERMITS={}
def engineering_permit(preparation):
    from .fg_s3_e_quality_foundation import synthetic_probe_permit
    synthetic_probe_permit(preparation)  # Only original built-in synthetic identity.
    if not preparation.receipt['identity']['source_f1']['engineering_only']:raise ValueError('F2_SYNTHETIC_F1_REQUIRED')
    permit=object();_PERMITS[id(preparation)]=(preparation,permit)
    return permit

class HeadsCounters:
    def __init__(self):
        self.state=dict(optimizer_updates=0,consumed_batches=0,skip_no_valid=0,skip_disabled_loss=0,
            target_positions=0,target_market_minutes=0,burnin_positions=0,burnin_market_minutes=0,
            head_valid_exposures=dict.fromkeys(('event','touch','adverse','score'),0),
            head_optimizer_exposures=dict.fromkeys(('event','touch','adverse','score'),0))
    def save(self):return copy.deepcopy(self.state)
    def load(self,state):
        if state.keys()!=self.state.keys():raise ValueError('F2_COUNTER_SCHEMA_MISMATCH')
        self.state=copy.deepcopy(state)

class HeadsRun:
    metrics_stage='f2'
    def __init__(self,preparation,index_path,root,*,resume=False,authorize_training=False,_probe=None):
        permit=_PERMITS.get(id(preparation))
        engineering=permit is not None and permit[0] is preparation and permit[1] is _probe
        if not engineering:require_training_authority(preparation,authorized=authorize_training)
        self.preparation=preparation;self.settings=s=preparation.settings
        if preflight(s).receipt['identity']!=preparation.receipt['identity']:raise ValueError('F2_RESUME_IDENTITY_MISMATCH')
        self.root=Path(root).resolve();self.index=Path(index_path).resolve()
        for src in ('corpus_root','source_f1_run'):
            source=Path(s[src]).resolve()
            if self.root.is_relative_to(source) or source.is_relative_to(self.root):raise ValueError('F2_RUN_OVERLAPS_SOURCE')
        self.identity=plain(dict(protocol='FG_S3_E_QUALITY_F2_RUN_V1',backend=BACKEND,
            preparation=preparation.receipt['identity'],index_sha256=digest(self.index),
            index_receipt_sha256=digest(self.index.with_suffix('.json')),code=execution_identity(),
            packages=runtime_versions(),controlled_small=engineering,training_admitted=not engineering,
            carry_rule='FRESH_PER_RANDOM_WINDOW',checkpoint_position='CONSUMED_BATCHES'))
        if resume:
            if read_manifest(self.root/'run_identity.json')['identity']!=self.identity:raise ValueError('F2_RESUME_IDENTITY_MISMATCH')
            probe=FullCheckpoint(self.root/'ckpt',elements.Counter(),self.identity,(),backend=BACKEND)
            selected=probe.latest()
            if selected is None:raise ValueError('F2_RESUME_CHECKPOINT_MISSING')
        elif self.root.exists():raise ValueError('F2_RUN_OUTPUT_MUST_BE_NEW')
        self.reader=EQualityCourseReader.open(preparation.source,stage='HEADS_VALID',sampling_spec=s['sampling_spec'],seed=s['model_init']['seed'],index_path=self.index)
        self.source=CourseStream(self.reader,s,adapter_cls=F2WindowAdapter)
        self.source.validate_snapshot()
        self.source.adapter.validate_support()
        if resume:self.source.validate_snapshot(pickle.loads((selected/'course_reader.pkl').read_bytes()))
        self.agent=create_agent(s,self.root/'model',controlled_small=engineering)
        if not resume:
            proof,state=inspect_f1(s,allow_engineering=engineering)
            receipt=transfer_parameters(self.agent,state,proof)
            receipt.pop('content_hash');receipt.update(f2_run_identity=self.identity,reader_seed=s['model_init']['seed'],sampling_spec=s['sampling_spec'])
            self.root.mkdir(parents=True)
            write_json(self.root/'run_identity.json',signed(dict(identity=self.identity)))
            write_json(self.root/'f1_to_f2_transfer.json',signed(receipt))
        self._initialize_runtime(resume,engineering)

    def _initialize_runtime(self,resume,engineering,*,counter_type=HeadsCounters):
        """Original shared offline loop/checkpoint lifecycle for F2 and F3."""
        s=self.settings
        self.stream=self.agent.stream(self.source)
        # Snapshot position advances for every consumed batch, even a skipped one.
        self.step=elements.Counter();self.counters=counter_type();self.samples=[0]
        self.schedule=StepSchedule(s['report_every_updates'],s['save_every_updates'])
        def set_carry(value):
            if value is not None:raise ValueError(self.metrics_stage.upper()+'_RANDOM_CARRY_MUST_BE_EMPTY')
        self.runtime=RuntimeState(self.schedule,SimpleNamespace(),[self.stream],lambda:None,set_carry,{},self.samples)
        self.cp=FullCheckpoint(self.root/'ckpt',self.step,self.identity,[self.stream],reserve_gib=1 if engineering else 10,keep_all=True,backend=BACKEND)
        self.cp.course_reader=self.source;self.cp.agent=self.agent;self.cp.step=self.step
        self.cp.counters=self.counters;self.cp.runtime=self.runtime
        self.metrics=[];self.last_batch=None
        if resume:
            self.cp.load()
            if int(self.step)!=self.counters.state['consumed_batches'] or int(self.agent.n_updates)!=self.counters.state['optimizer_updates']:raise ValueError(self.metrics_stage.upper()+'_RESTORED_COUNTER_MISMATCH')
            reconcile_update_log(self.root,int(self.step),0,position_key='consumed_batches')
        else:self.cp.save()

    def update(self):
        self._check_consumption_budget()
        if int(self.agent.n_updates)>=self.settings['optimizer_updates']:raise ValueError(self.metrics_stage.upper()+'_UPDATE_BUDGET_EXHAUSTED')
        self.runtime.start();batch=next(self.stream);self.last_batch=host(batch)
        try:
            _,_,metrics=self.agent.train(self.agent.init_train(self.agent.config.batch_size),batch)
        except FloatingPointError as error:
            self._reject_batch(error)
            raise
        prefix=self.metrics_stage+'/'
        metrics=host(metrics);updated=bool(metrics[prefix+'updated']);self._consume_batch_boundary();c=self.counters.state
        c['consumed_batches']+=1;c['optimizer_updates']+=int(updated)
        c['skip_no_valid']+=int(bool(metrics[prefix+'skip_no_valid']))
        c['skip_disabled_loss']+=int(not updated and not metrics[prefix+'skip_no_valid'])
        for k in ('target_positions','target_market_minutes','burnin_positions','burnin_market_minutes'):c[k]+=int(metrics[prefix+k])
        self._accumulate_metrics(metrics,updated)
        self.samples[0]+=int(metrics[prefix+'target_positions'])
        if int(self.agent.n_updates)!=c['optimizer_updates']:raise ValueError(self.metrics_stage.upper()+'_OPTIMIZER_COUNT_DRIFT')
        report,save=self._schedule_update(updated)
        row={k:np.asarray(v).tolist() for k,v in metrics.items()};row[prefix+'report_boundary']=report
        self.metrics.append(row)
        with (self.root/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(local_update=c['optimizer_updates'],agent_update=int(self.agent.n_updates),consumed_batches=c['consumed_batches'],metrics=row))+'\n')
        if save or not updated:self._save_checkpoint()
        return row

    def _consume_batch_boundary(self):self.step.increment()
    def _schedule_update(self,updated):
        return self.schedule.advance(self.counters.state['optimizer_updates']) if updated else (False,False)
    def _save_checkpoint(self):return self.cp.save()

    def _accumulate_metrics(self,metrics,updated):
        c=self.counters.state
        for name in c['head_valid_exposures']:
            valid=int(metrics['quality/'+name+'_valid_count']);c['head_valid_exposures'][name]+=valid
            if updated and metrics['f2/quality_'+name+'_active']:c['head_optimizer_exposures'][name]+=valid

    def _reject_batch(self,error):
        import uuid
        write_json(self.root/('rejected_'+uuid.uuid4().hex+'.json'),signed(dict(error=str(error),
            source=self.preparation.source.receipt['content_identity'],stepid=self.last_batch['stepid'].tolist(),
            consumed_before=int(self.step),optimizer_updates=int(self.agent.n_updates))))

    def _check_consumption_budget(self):pass

    def run(self,stop_at=None):
        stop=self.settings['optimizer_updates'] if stop_at is None else stop_at
        if type(stop) is not int or not int(self.agent.n_updates)<=stop<=self.settings['optimizer_updates']:raise ValueError(self.metrics_stage.upper()+'_STOP_OUTSIDE_UPDATE_BUDGET')
        while int(self.agent.n_updates)<stop:self.update()
        self.cp.save(final=True)
        return self.result()

    def result(self):
        return dict(status='F2_ENGINEERING_COMPLETE' if self.identity['controlled_small'] else 'F2_TRAIN_AUTHORIZED',
            foundation_status=self.preparation.receipt['identity']['source_f1']['foundation_status'],
            counters=self.counters.save(),training_admitted=self.identity['training_admitted'],model_research_go=False,checkpoint=str(self.cp.latest()))
