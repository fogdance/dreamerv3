"""S3-C new-run transfer and exact full-state resume, using original components."""
from functools import partial
from pathlib import Path
import copy
import hashlib
import json
import pickle
import subprocess
import uuid

import elements
import embodied
import jax
import numpy as np
import cloudpickle

from embodied.run.action_mask_warmup import ActionMaskWarmup,METRICS
from embodied.run.fg_checkpoint import FullCheckpoint,RuntimeState,StepSchedule,file_hash,plain,validate_bundle
from embodied.run.fg_s3_source_admission import inspect_source,transfer_control_state,runtime_versions
from embodied.run.fg_s3_source_transfer import make_agent,apply_and_audit_transfer
from embodied.run.fg_s3c_stream import TeacherJoinStream
from embodied.run.fg_s3_teacher_adapter import admit_replay_directory
from dreamerv3.fg_s3c_agent import S3CAgent
from gym_trading_env.research.fg_replay import read_manifest,signed,write_json
from gym_trading_env.research.fg_s3_teacher_catalog import load_catalog,guard_output

PROTOCOL='FG_SINGLE3_S3C1_CONCURRENT_BC_V3'


def code_identity():
    root=Path(__file__).resolve().parents[2]
    files=[root/'dreamerv3/configs.yaml']
    for package in (root/'dreamerv3',root/'embodied',root.parent/'gym-trading-env/src/gym_trading_env'):
        files.extend(p for p in package.rglob('*.py') if 'tests' not in p.parts)
    return {str(p):file_hash(p) for p in sorted(files)}


def saved_run(run_dir,catalog):
    """Verify own immutable run artifacts without reading a model source again."""
    root=Path(run_dir)
    identity=read_manifest(root/'run_identity.json')['identity']
    if identity['protocol']!=PROTOCOL:
        raise ValueError('Foreign S3-C run protocol')
    if identity['code_sha256']!=code_identity() or identity['catalog_sha256']!=catalog.manifest_sha256:
        raise ValueError('Resume code/Catalog identity changed')
    if identity['packages']!=runtime_versions():raise ValueError('Resume package identity changed')
    for name,key,label in (
        ('source_admission_receipt.json','source_admission_file_sha256','source receipt'),
        ('source_transfer_receipt.json','source_transfer_file_sha256','transfer receipt'),
        ('source_control.pkl','source_control_file_sha256','source control'),
        ('teacher_catalog_identity.json','teacher_catalog_file_sha256','teacher identity'),
        ('run_settings.json','settings_file_sha256','run settings')):
        if file_hash(root/name)!=identity[key]:raise ValueError('Resume '+label+' changed')
    settings=read_manifest(root/'run_settings.json')
    from dreamerv3.configuration import read_config_yaml
    if read_config_yaml(root/'config.yaml')!=settings['config']:
        raise ValueError('Resume resolved config changed')
    source=read_manifest(root/'source_admission_receipt.json')
    if source['catalog_manifest_sha256']!=catalog.manifest_sha256 or source['kind']!=identity['source_kind']:
        raise ValueError('Resume source/Catalog contract changed')
    return identity,settings


def audit_checkpoint(run_dir,*,catalog):
    """Audit a real full bundle, including model and update counters, on CPU."""
    root=Path(run_dir);cat=load_catalog(catalog)
    identity,_=saved_run(root,cat)
    name=(root/'ckpt/latest').read_text().strip()
    if not name or Path(name).name!=name:raise ValueError('Checkpoint pointer escaped run')
    node=root/'ckpt'/name;meta=validate_bundle(node,identity)
    state=pickle.loads((node/'agent.pkl').read_bytes())
    counters=pickle.loads((node/'s3c_counters.pkl').read_bytes())
    if (state['counters']['updates']!=identity['source_update_offset']+meta['step']
            or counters['all_updates']!=meta['step']):
        raise ValueError('Checkpoint source/local/model update counters disagree')
    if any('s3a_bc_opt' in k or 's3b_bc_opt' in k for k in state['params']):
        raise ValueError('Independent BC optimizer in full Agent')
    if not any(k.startswith('opt/') for k in state['params']) or not all(np.isfinite(v).all() for v in state['params'].values()):
        raise ValueError('Checkpoint joint optimizer missing or Agent nonfinite')
    counts=JointCounters();counts.load(counters)
    return dict(status='TRAINING_CHECKPOINT_VERIFIED',checkpoint=str(node),
                checkpoint_manifest_sha256=file_hash(node/'manifest.json'),identity=identity,
                local_updates=meta['step'],agent_updates=state['counters']['updates'],counters=counts.metrics())


class JointCounters:
    def __init__(self):
        self.state=dict(all_updates=0,wm_updates=0,gate_ready_updates=0,bc_effective_updates=0,
            bc_valid_rows_seen=0,bc_add_rows_seen=0,bc_wait_rows_seen=0,no_bc_batch_updates=0,
            tp=0,fp=0,fn=0,tn=0)

    def observe(self,metrics,batch,weight,prefix):
        valid=np.asarray(batch['teacher_valid'][:,prefix:]);y=np.asarray(batch['teacher_action'][:,prefix:])
        count=int(valid.sum());adds=int((valid&(y==1)).sum());gate=bool(metrics['avail/ready'])
        if weight>0 and (int(metrics['bc/valid_count'])!=count or bool(metrics['bc/gate_ready'])!=gate):
            raise ValueError('BC metrics do not match consumed batch/Gate')
        s=self.state;s['all_updates']+=1;s['wm_updates']+=1;s['gate_ready_updates']+=int(gate)
        s['bc_effective_updates']+=int(gate and weight>0 and count>0)
        s['bc_valid_rows_seen']+=count;s['bc_add_rows_seen']+=adds;s['bc_wait_rows_seen']+=count-adds
        s['no_bc_batch_updates']+=int(count==0)
        for key in ('tp','fp','fn','tn'):s[key]+=int(metrics.get('bc/'+key,0))

    def save(self):return copy.deepcopy(self.state)

    def load(self,state):self.state=copy.deepcopy(state)

    def metrics(self):
        s=self.state;denom=max(s['bc_valid_rows_seen'],1)
        return dict(s,zero_bc_batch_ratio=s['no_bc_batch_updates']/max(s['all_updates'],1),
            always_wait_accuracy=s['bc_wait_rows_seen']/denom,
            classification_rows=s['tp']+s['fp']+s['fn']+s['tn'],
            accuracy=(s['tp']+s['tn'])/(s['tp']+s['fp']+s['fn']+s['tn']) if s['tp']+s['fp']+s['fn']+s['tn'] else None,
            precision=s['tp']/(s['tp']+s['fp']) if s['tp']+s['fp'] else None,
            recall=s['tp']/(s['tp']+s['fn']) if s['tp']+s['fn'] else None)


class S3CRun:
    def __init__(self,run_dir,*,catalog,config=None,source_kind=None,source_checkpoint=None,
                 base_checkpoint=None,features=None,seed=None,bc_lambda=None,replay_seed=202,
                 replay_directory=None,capacity=None,consec=1,report_every=100,save_every=100,
                 resume=False,device='cpu',diagnostic=False,target_updates=None,managed_binding=None,
                 expected_source_hash=None):
        root=Path(run_dir).resolve();self.root=root
        self.catalog=cat=load_catalog(catalog)
        self.weight=bc_lambda
        if resume:
            if any(x is not None for x in (source_kind,source_checkpoint,base_checkpoint,features,seed,config,managed_binding,expected_source_hash)):
                raise ValueError('Resume forbids a new source transfer')
            identity,saved=saved_run(root,cat)
            if bc_lambda is not None and bc_lambda!=identity['bc_lambda']:
                raise ValueError('Resume BC lambda differs')
            if target_updates is not None and target_updates!=saved['target_updates']:
                raise ValueError('Resume update budget differs')
            config=saved['config'];self.weight=identity['bc_lambda'];consec=saved['consec']
            report_every=saved['report_every'];save_every=saved['save_every'];capacity=saved['capacity']
            target_updates=saved['target_updates']
            replay_seed=identity['replay_seed'];control=pickle.loads((root/'source_control.pkl').read_bytes());offset=identity['source_update_offset']
        else:
            if source_kind is None or bc_lambda is None or not np.isfinite(bc_lambda) or bc_lambda<0:
                raise ValueError('New run requires explicit source and finite nonnegative lambda')
            spec=inspect_source(source_kind,catalog=cat,path=source_checkpoint,seed=seed,config=config,
                                base_checkpoint=base_checkpoint,features=features,diagnostic=diagnostic)
            if expected_source_hash is not None and spec['content_hash']!=expected_source_hash:
                raise ValueError('Source changed after immutable preparation')
            if spec['origin']['diagnostic'] and device!='cpu':
                raise ValueError('Diagnostic sources require CPU')
            cfg=plain(config or spec['origin']['config'])
            if source_kind!='direct_init' and cfg['agent']!=spec['origin']['config']['agent']:
                raise ValueError('Target changed source network/optimizer config')
            config=cfg;config['batch_length']=64
            control=transfer_control_state(spec)
            if not control['complete'] and not spec['origin']['diagnostic']:
                raise ValueError('Formal transfer missing Gate controller/window state')
            protected=[cat.root,Path(cat.manifest['source']['dataset']),Path(cat.manifest['source']['oracle_receipt']),Path(cat.manifest['labels'])]
            for p in (source_checkpoint,base_checkpoint):
                if p:protected.append(Path(p).resolve().parent.parent)
            guard_output(root,protected);root.mkdir(parents=True)
        self.config=cfg=elements.Config(config)
        self.target_updates=target_updates
        if target_updates is not None and (type(target_updates) is not int or target_updates<1):
            raise ValueError('Target budget must be positive gradient updates')
        if cfg.replay_context!=1 or cfg.agent.replay_context_mode!='recompute_prefix' or consec<1:
            raise ValueError('S3-C requires one recomputed prefix and positive consec')
        self.agent=agent=make_agent(plain(cfg),root,agent_cls=S3CAgent,bc_lambda=self.weight,device=device)
        if not resume:
            receipt=apply_and_audit_transfer(agent,spec);offset=receipt['source_update_offset']
            write_json(root/'source_admission_receipt.json',spec)
            write_json(root/'source_transfer_receipt.json',receipt)
            (root/'source_control.pkl').write_bytes(cloudpickle.dumps(control,protocol=5))
        self.offset=offset
        dataset=Path(cat.manifest['source']['dataset']);dm=read_manifest(dataset/'manifest.json')
        if capacity is None:
            capacity=sum(x['sequence_starts'] for x in dm['traces'] if x['candidate']=='C1' and x['policy_id'] in ('P1','P2a','P2b','P3','P4'))
        length=consec*cfg.batch_length+cfg.replay_context
        if int(capacity)<cfg.batch_size*length:raise ValueError('Replay capacity below one batch')
        self.replay=replay=embodied.replay.Replay(length=length,capacity=int(capacity),chunksize=1024,
            directory=root/'replay',online=False,seed=replay_seed)
        if not resume:
            directory=Path(replay_directory) if replay_directory else dataset/dm['views']['C1_natural_oracle']
            # Directory may select a diagnostic subset, but every chunk must be
            # authenticated by the same shared Catalog. Check before Replay.load,
            # which otherwise tolerates corrupt/foreign files.
            admit_replay_directory(directory,cat,diagnostic=spec['origin']['diagnostic'])
            replay.load(directory=directory)
            if len(replay)<cfg.batch_size*length:raise ValueError('Replay too small for diagnostic/formal batch')
        def stream(mode):
            base=embodied.streams.Consec(embodied.streams.Stateless(partial(replay.sample,cfg.batch_size,mode)),
                length=cfg.batch_length if mode=='train' else cfg.report_length,
                consec=consec if mode=='train' else 1,prefix=cfg.replay_context,strict=mode=='train')
            return agent.stream(TeacherJoinStream(base,cat,mode=mode))
        self.streams=(stream('train'),stream('report'))
        self.carry=dict(train=agent.init_train(cfg.batch_size),report=agent.init_report(cfg.batch_size))
        self.aggregates={k:elements.Agg() for k in ('train','report','warmup')}
        self.samples=[control['samples']]
        self.warmup=ActionMaskWarmup(cfg.action_mask_warmup,False)
        if control['opening']:
            self.warmup.opening_thresholds=dict(control['opening']['thresholds'])
            self.warmup.required_passes=control['opening']['consecutive_reports']
        if control['gate']:self.warmup.load(control['gate'])
        else:self.warmup.actor_enabled=agent.get_avail_actor_enabled()
        if control['aggregate']:self.aggregates['warmup'].reducers=copy.deepcopy(control['aggregate'])
        self.step=elements.Counter();self.counters=JointCounters()
        self.schedule=StepSchedule(report_every,save_every)
        def set_carry(state):self.carry={k:agent.restore_carry(v) for k,v in state.items()}
        self.runtime=RuntimeState(self.schedule,elements.when.Ratio(1),self.streams,
            lambda:self.carry,set_carry,self.aggregates,self.samples)
        if not resume:
            write_json(root/'run_settings.json',signed(dict(config=plain(cfg),consec=consec,
                capacity=int(capacity),report_every=report_every,save_every=save_every,
                target_updates=target_updates,budget_unit='gradient_updates')))
            write_json(root/'teacher_catalog_identity.json',signed(dict(
                protocol=cat.manifest['protocol'],path=str(cat.root),manifest_sha256=cat.manifest_sha256,
                dataset_sha256=cat.manifest['source']['dataset_sha256'],oracle_sha256=cat.manifest['source']['oracle_sha256'])))
            identity=dict(protocol=PROTOCOL,run_id=managed_binding['run_id'] if managed_binding else uuid.uuid4().hex,source_kind=source_kind,
                source_update_offset=offset,source_bc_updates=spec['origin'].get('bc_updates',0),
                source_admission_file_sha256=file_hash(root/'source_admission_receipt.json'),
                source_transfer_file_sha256=file_hash(root/'source_transfer_receipt.json'),
                source_control_file_sha256=file_hash(root/'source_control.pkl'),
                settings_file_sha256=file_hash(root/'run_settings.json'),
                teacher_catalog_file_sha256=file_hash(root/'teacher_catalog_identity.json'),
                managed_preparation=managed_binding,target_updates=target_updates,budget_unit='gradient_updates',
                catalog_sha256=cat.manifest_sha256,catalog_protocol=cat.manifest['protocol'],
                source_admission_protocol=spec['protocol'],dataset_sha256=cat.manifest['source']['dataset_sha256'],
                reward_params_hash=cat.manifest['source']['reward_params_hash'],obs_version=cat.manifest['obs_version'],
                observation_identity=cat.manifest['observation_identity'],training_fold='TRAIN',pool='C1_natural_oracle',
                oracle_sha256=cat.manifest['source']['oracle_sha256'],oracle_optimality='exact_proven',
                bc_type='equal_weight_P4',bc_lambda=self.weight,model_seed=cfg.seed,replay_seed=replay_seed,
                optimizer_transfer_policy=spec['optimizer_transfer_policy'],code_sha256=code_identity(),
                source_versions={str(repo):subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
                    for repo in (Path(__file__).resolve().parents[2],Path(__file__).resolve().parents[3]/'gym-trading-env')},
                network_config_sha256=hashlib.sha256(json.dumps(plain(cfg.agent),sort_keys=True).encode()).hexdigest(),
                optimizer_config_sha256=hashlib.sha256(json.dumps(plain(cfg.agent.opt),sort_keys=True).encode()).hexdigest(),
                packages=runtime_versions(),diagnostic=spec['origin']['diagnostic'],
                action_mask_gate_policy=dict(config=plain(cfg.action_mask_warmup),
                    opening_thresholds=self.warmup.opening_thresholds,
                    consecutive_reports=self.warmup.required_passes,automatic_activation=self.warmup.enabled),
                replay_context=1,replay_context_mode='recompute_prefix',sampling_flags='embodied_native_sample',
                learning_config_sha256=hashlib.sha256(json.dumps(plain(cfg),sort_keys=True).encode()).hexdigest())
            write_json(root/'run_identity.json',signed(dict(identity=identity)))
            import yaml
            (root/'config.yaml').write_text(yaml.safe_dump(plain(cfg)))
        self.identity=identity
        self.cp=FullCheckpoint(root/'ckpt',self.step,identity,self.streams,reserve_gib=1,keep_all=True)
        self.cp.agent=agent;self.cp.replay=replay;self.cp.runtime=self.runtime;self.cp.step=self.step
        self.cp.action_mask_warmup=self.warmup;self.cp.s3c_counters=self.counters
        if resume:
            self.cp.load()
            if int(agent.n_updates)!=offset+int(self.step) or self.counters.state['all_updates']!=int(self.step):
                raise ValueError('Restored source offset/local counters disagree')
        else:self.cp.save()

    def update(self):
        if self.target_updates is not None and int(self.step)>=self.target_updates:
            raise ValueError('Configured gradient update budget is exhausted')
        self.runtime.start()
        if self.agent.get_avail_actor_enabled()!=self.warmup.actor_enabled:
            raise ValueError('Gate model/controller disagree')
        batch=next(self.streams[0]);ids=np.array(jax.device_get(batch['stepid']),copy=True)
        teacher=np.array(jax.device_get(batch['teacher_action']),copy=True)
        valid=np.array(jax.device_get(batch['teacher_valid']),copy=True)
        self.carry['train'],outs,mets=self.agent.train(self.carry['train'],batch)
        if not bool(mets['opt/finite']):raise FloatingPointError('Joint model update nonfinite')
        # The synchronous offline wrapper does not enqueue a policy refresh.
        # Publish current WM/Actor before an exact checkpoint or inference.
        self.agent.sync_policy_from_train()
        if 'replay' in outs:self.replay.update(outs['replay'])
        self.counters.observe(mets,dict(teacher_action=teacher,teacher_valid=valid),self.weight,1)
        self.aggregates['train'].add(mets)
        if self.warmup.enabled:
            self.aggregates['warmup'].add({k:mets[k] for k,_ in METRICS.values() if k in mets})
            for prefix in ('post','prior'):
                for suffix in ('add_positives','add_true_positives'):
                    key='avail/'+prefix+'_'+suffix
                    self.aggregates['warmup'].add(key,mets[key],agg='sum')
            self.samples[0]+=self.config.batch_size*self.config.batch_length
        self.step.increment()
        if int(self.agent.n_updates)!=self.offset+int(self.step):raise ValueError('Joint update counter drift')
        report,save=self.schedule.advance(self.step)
        if report:
            self.aggregates['report'].reset()
            rb=next(self.streams[1]);self.carry['report'],rm=self.agent.report(self.carry['report'],rb)
            self.aggregates['report'].add(rm)
            if self.warmup.ready(self.samples[0]):
                wm=self.aggregates['warmup'].result();self.samples[0]=0
                for prefix in ('post','prior'):
                    positive=wm.get('avail/'+prefix+'_add_positives',0)
                    wm['avail/'+prefix+'_add_recall']=wm.get('avail/'+prefix+'_add_true_positives',0)/positive if positive else float('nan')
                self.warmup.update(wm,self.offset+int(self.step),allow_activation=True)
                self.agent.set_avail_actor_enabled(self.warmup.actor_enabled)
        row=dict(local_update=int(self.step),agent_update=int(self.agent.n_updates),
            bc_lambda=self.weight,gate_used=bool(mets['avail/ready']),counters=self.counters.metrics(),
            gate_after_report=self.agent.get_avail_actor_enabled(),
            gate_controller=self.warmup.metrics(),
            stepid_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            teacher_sha256=hashlib.sha256(teacher.tobytes()+valid.tobytes()).hexdigest(),
            metrics={k:float(np.asarray(v)) for k,v in mets.items() if np.asarray(v).size==1 and np.isfinite(v).all()})
        with (self.root/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if save:self.cp.save()
        return dict(stepid=ids,teacher_action=teacher,teacher_valid=valid,metrics=mets)

    def train_to(self,target):
        if isinstance(target,bool) or int(target)!=target or target<int(self.step):raise ValueError('Invalid target update budget')
        if self.target_updates is not None and target>self.target_updates:raise ValueError('Target exceeds frozen update budget')
        while int(self.step)<target:self.update()
        self.cp.save(final=True)
        checkpoint=self.cp.latest()
        receipt=signed(dict(status='CPU_DIAGNOSTIC_COMPLETE' if self.identity['diagnostic'] else 'TRAIN_COMPLETE',
            identity=self.identity,local_updates=int(self.step),source_update_offset=self.offset,
            agent_updates=int(self.agent.n_updates),counters=self.counters.metrics(),
            checkpoint=str(checkpoint),checkpoint_manifest_sha256=file_hash(checkpoint/'manifest.json'),
            devices=[str(x) for x in jax.devices()],dev_evaluated=False,test_evaluated=False,
            interpretation='TRAIN replay imitation and engineering evidence only'))
        write_json(self.root/'training_receipt.json',receipt)
        return receipt
