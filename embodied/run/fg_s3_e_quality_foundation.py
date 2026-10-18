"""F1 offline coordination; original Reader, Agent, optimizer and FullCheckpoint.

No Env collector, teacher objective, policy training or independent Replay.
"""
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
import elements
import numpy as np
from .fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,host,plain
from .fg_s3_source_admission import runtime_versions
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import digest,read_manifest,signed,write_json
from gym_trading_env.research.fg_s3_e_quality_foundation import (
    F1WindowAdapter,create_agent,require_training_authority,execution_identity)
from gym_trading_env.research.fg_s3_e_quality_supervision import inspect_source
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader

BACKEND='external_immutable_course_reader'

class CourseStream:
    """Lifecycle adapter, not another sampler or Replay."""
    def __init__(self,reader,settings):
        self.reader=reader;self.adapter=F1WindowAdapter(reader,settings)
    def __iter__(self):return self
    def __next__(self):return self.adapter.sample()[0]
    def validate_snapshot(self,state=None):
        inspect_source(self.reader.admission.source)  # Rehash every bound member.
        if digest(self.reader.index_path)!=self.reader.index_hash:
            raise ValueError('F1_EXTERNAL_INDEX_CHANGED')
        current=self.reader.state_dict()
        if state is not None and any(state.get(k)!=current[k] for k in
                ('protocol','source','index','stage','sampling_spec')):
            raise ValueError('F1_EXTERNAL_READER_IDENTITY_MISMATCH')
    def save(self):
        # Called once per prefetched batch. Reader's stat guards/cache are the
        # hot path; full hashing is reserved for checkpoint save/load.
        return self.reader.state_dict()
    save_full=save
    def load(self,state):
        self.validate_snapshot(state);self.reader.load_state_dict(state)
    load_full=load

class Counters:
    def __init__(self):
        self.state=dict(optimizer_updates=0,target_positions=0,target_market_minutes=0,
                       burnin_positions=0,burnin_market_minutes=0,consumed_batches=0)
    def save(self):return copy.deepcopy(self.state)
    def load(self,state):
        if set(state)!=set(self.state):raise ValueError('F1_COUNTER_SCHEMA_MISMATCH')
        self.state=copy.deepcopy(state)

# An engineering probe passes an in-process capability, never an editable
# "synthetic:true" setting. The public CLI builds its fixed source itself.
_PROBE=object()

def synthetic_probe_permit(preparation):
    """Only the fixed, visibly non-certified engineering fixture identity."""
    root=Path(preparation.settings['corpus_root'])
    corpus=read_manifest(root/'corpus_manifest.json')
    import json
    course=json.loads((root/'course_v2.json').read_text())
    lock=read_manifest(root/'source_lock.json')
    if (corpus.get('e_profile_identity')!='c'*64 or lock.get('contract')!='CZCE.FG2701'
            or course['data_identity']['market_sources']!=[dict(contract='CZCE.FG2701',
                source_sha256='c'*64,env_config_sha256='d'*64,metadata_sha256='e'*64)]):
        raise ValueError('F1_BUILTIN_SYNTHETIC_IDENTITY_REQUIRED')
    return (_PROBE,preparation.source.receipt['content_identity'])

class FoundationRun:
    def __init__(self,preparation,index_path,root,*,resume=False,authorize_training=False,
                 _probe=None):
        engineering=(type(_probe) is tuple and len(_probe)==2 and _probe[0] is _PROBE
                     and _probe[1]==preparation.source.receipt['content_identity'])
        if not engineering:require_training_authority(preparation,authorized=authorize_training)
        self.preparation=preparation;self.settings=s=preparation.settings
        from gym_trading_env.research.fg_s3_e_quality_foundation import preflight
        if preflight(s).receipt['identity']!=preparation.receipt['identity']:
            raise ValueError('F1_RESUME_IDENTITY_MISMATCH: preparation settings/content changed')
        self.root=Path(root).resolve();self.index=Path(index_path).resolve()
        if self.root.is_relative_to(Path(s['corpus_root']).resolve()):
            raise ValueError('F1_RUN_CANNOT_MODIFY_CORPUS')
        # Fail before run mkdir/model creation when a resume identity differs.
        self.identity=dict(protocol='FG_S3_E_QUALITY_F1_RUN_V1',backend=BACKEND,
            preparation=preparation.receipt['identity'],index_sha256=digest(self.index),
            index_receipt_sha256=digest(self.index.with_suffix('.json')),
            code=execution_identity(),packages=runtime_versions(),
            controlled_small=engineering,carry_rule='FRESH_PER_RANDOM_WINDOW',
            training_admitted=not engineering)
        from gym_trading_env.research.fg_s3_e_quality_foundation import model_config
        from dreamerv3.fg_s3_e_quality_foundation import model_extensions
        self.identity['effective_model']=model_config(s,controlled_small=engineering)
        self.identity['model_extension']=model_extensions(s['target_steps'],s['burnin_steps'])
        self.identity=plain(self.identity)
        if resume:
            saved=read_manifest(self.root/'run_identity.json')['identity']
            if saved!=self.identity:raise ValueError('F1_RESUME_IDENTITY_MISMATCH')
            if not (self.root/'ckpt').is_dir():raise ValueError('F1_RESUME_CHECKPOINT_MISSING')
        elif self.root.exists():raise ValueError('F1_RUN_OUTPUT_MUST_BE_NEW')
        self.reader=EQualityCourseReader.open(preparation.source,stage='WM_NATURAL',
            sampling_spec=s['sampling_spec'],seed=s['model_init']['seed'],index_path=self.index)
        self.source=CourseStream(self.reader,s);self.source.validate_snapshot()
        if resume:
            probe=FullCheckpoint(self.root/'ckpt',elements.Counter(),self.identity,(),backend=BACKEND)
            selected=probe.latest()
            if selected is None:raise ValueError('F1_RESUME_CHECKPOINT_MISSING')
            import pickle
            self.source.validate_snapshot(pickle.loads((selected/'course_reader.pkl').read_bytes()))
        self.root.mkdir(parents=True,exist_ok=resume)
        self.agent=create_agent(s,self.root/'model',controlled_small=engineering)
        if not resume:write_json(self.root/'run_identity.json',signed(dict(identity=self.identity)))
        self.stream=self.agent.stream(self.source)
        self.step=elements.Counter();self.counters=Counters();self.samples=[0]
        self.schedule=StepSchedule(s['report_every_updates'],s['save_every_updates'])
        # No cross-window carry. RuntimeState still saves all RNGs, the actual
        # original Prefetch queue, original seed counter and report schedule.
        def set_carry(value):
            if value is not None:raise ValueError('F1_RANDOM_CARRY_MUST_BE_EMPTY')
        self.runtime=RuntimeState(self.schedule,SimpleNamespace(),[self.stream],
                                  lambda:None,set_carry,{},self.samples)
        self.cp=FullCheckpoint(self.root/'ckpt',self.step,self.identity,[self.stream],
                              reserve_gib=1,keep_all=True,backend=BACKEND)
        self.cp.course_reader=self.source;self.cp.agent=self.agent
        self.cp.step=self.step;self.cp.counters=self.counters;self.cp.runtime=self.runtime
        self.metrics=[];self.last_batch=None
        if resume:
            self.cp.load()
            if int(self.agent.n_updates)!=int(self.step) or int(self.step)!=self.counters.state['optimizer_updates']:
                raise ValueError('F1_RESTORED_UPDATE_COUNTER_MISMATCH')
            from .fg_s3c_train import reconcile_update_log
            reconcile_update_log(self.root,int(self.step),0)
        else:self.cp.save()

    def update(self):
        if int(self.step)>=self.settings['optimizer_updates']:raise ValueError('F1_UPDATE_BUDGET_EXHAUSTED')
        self.runtime.start();batch=next(self.stream);self.last_batch=host(batch)
        _,_,metrics=self.agent.train(self.agent.init_train(self.settings['batch_size']),batch)
        metrics=host(metrics)
        if not bool(metrics['f1/finite']):raise FloatingPointError('F1_NONFINITE_UPDATE_REJECTED')
        self.step.increment();c=self.counters.state;c['optimizer_updates']+=1;c['consumed_batches']+=1
        for k in ('target_positions','target_market_minutes','burnin_positions','burnin_market_minutes'):
            c[k]+=int(metrics['f1/'+k])
        self.samples[0]+=int(metrics['f1/target_positions'])
        if int(self.agent.n_updates)!=int(self.step):raise ValueError('F1_OPTIMIZER_COUNT_DRIFT')
        report,save=self.schedule.advance(int(self.step))
        row={k:np.asarray(v).tolist() for k,v in metrics.items()}
        row['f1/report_boundary']=report
        self.metrics.append(row)
        # Append-only diagnostics; authoritative history is checkpoint state.
        import json
        with (self.root/'updates.jsonl').open('a') as f:
            f.write(json.dumps(dict(local_update=int(self.step),agent_update=int(self.agent.n_updates),metrics=row))+'\n')
        if save:self.cp.save()
        return row

    def run(self,stop_at=None):
        stop=self.settings['optimizer_updates'] if stop_at is None else stop_at
        if type(stop) is not int or not int(self.step)<=stop<=self.settings['optimizer_updates']:
            raise ValueError('F1_STOP_OUTSIDE_UPDATE_BUDGET')
        while int(self.step)<stop:self.update()
        self.cp.save(final=True)
        return dict(status='F1_ENGINEERING_COMPLETE' if self.identity['controlled_small'] else 'F1_TRAIN_AUTHORIZED',
                    counters=self.counters.save(),training_admitted=self.identity['training_admitted'],
                    model_research_go=False,checkpoint=str(self.cp.latest()))
