"""Uniform-row pure BC loop, independent counters, deterministic full resume."""
from pathlib import Path
import json
import uuid

import numpy as np

from embodied.run.fg_checkpoint import file_hash
from embodied.run.fg_s3a_checkpoint import (PROTOCOL,code_identity,package_versions,load_source,restore_bc,save_bc)
from dreamerv3.fg_s3a_actor import ActorBC,DEFAULT_OPT,classification_metrics
from gym_trading_env.research.fg_replay import read_manifest,signed,write_json


class Sampler:
    def __init__(self,rows,seed):
        if rows<1:raise ValueError('No supervised rows')
        self.rows=rows;self.rng=np.random.default_rng(seed);self.order=self.rng.permutation(rows)
        self.cursor=0;self.epoch=0

    def next(self,size):
        if size<1:raise ValueError('Batch size must be positive')
        result=[];remaining=size
        while remaining:
            take=min(remaining,self.rows-self.cursor)
            result.append(self.order[self.cursor:self.cursor+take]);self.cursor+=take;remaining-=take
            if self.cursor==self.rows:
                self.order=self.rng.permutation(self.rows);self.cursor=0;self.epoch+=1
        return np.concatenate(result)

    def save(self):
        return dict(rows=self.rows,rng=self.rng.bit_generator.state,order=self.order.copy(),cursor=self.cursor,epoch=self.epoch)

    def load(self,state):
        if (state['rows']!=self.rows or not np.array_equal(np.sort(state['order']),np.arange(self.rows))
                or not 0<=state['cursor']<self.rows):raise ValueError('Invalid BC sampler checkpoint')
        self.rng.bit_generator.state=state['rng'];self.order=state['order'].copy()
        self.cursor=state['cursor'];self.epoch=state['epoch']


def training_identity(source,cache,seed,run_id,batch_size,opt_config):
    return dict(protocol=PROTOCOL,source_manifest_sha256=source['manifest_sha256'],
        source_identity_hash=source['identity_hash'],diagnostic=source['diagnostic'],
        feature_manifest_sha256=file_hash(cache.root/'manifest.json'),
        label_manifest_sha256=cache.manifest['labels_manifest_sha256'],seed=seed,run_id=run_id,
        batch_size=batch_size,optimizer=dict(DEFAULT_OPT,**(opt_config or {})),
        sampler='uniform_permutation_all_legal_rows_no_class_reweighting',code_sha256=code_identity(),
        packages=package_versions())


def evaluate_classifier(bc,cache):
    probabilities=[];labels=[];years=[];contracts=[]
    for shard,x,a in zip(cache.manifest['shards'],cache.features,cache.index):
        p=[]
        for start in range(0,len(x),256):p.append(bc.probabilities(x[start:start+256],a['mask'][start:start+256]))
        probabilities.append(np.concatenate(p));labels.append(a['label']);years.append(a['year'])
        contracts.extend([shard['contract']]*len(x))
    p=np.concatenate(probabilities);y=np.concatenate(labels);year=np.concatenate(years);contract=np.asarray(contracts)
    strata={}
    for name,group in [('contract',contract),('year',year)]:
        for value in np.unique(group):
            select=group==value;strata[name+'/'+str(value)]=classification_metrics(y[select],p[select])
    return dict(classification_metrics(y,p),strata=strata,
                interpretation='TRAIN Oracle imitation only; correlated rows; no independent validation')


def train(source,cache,output,*,seed=0,run_id='bc',batch_size=256,updates=10,
          opt_config=None,resume=False,save_every=100,device='cpu',agent=None):
    if updates<1 or batch_size<1 or save_every<1:raise ValueError('Positive BC budgets required')
    root=Path(output).resolve()
    if root.is_relative_to(Path(source['path'])) or root.is_relative_to(cache.root) or root.is_relative_to(Path(cache.manifest['labels'])):
        raise ValueError('BC output cannot modify source/cache directories')
    identity=training_identity(source,cache,seed,run_id,batch_size,opt_config)
    if resume:
        existing=read_manifest(root/'run_identity.json')
        if existing['identity']!=identity:raise ValueError('BC resume run identity mismatch')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/'run_identity.json',signed(dict(identity=identity)))
    agent=agent or load_source(source,root,device=device)
    bc=ActorBC(agent,opt_config);sampler=Sampler(cache.rows,seed)
    if bc.source_digests!=cache.manifest['source_param_digests']:
        raise ValueError('Loaded source parameters do not match feature-producing source')
    ckpt=root/'ckpt'
    if resume:
        name=(ckpt/'latest').read_text().strip()
        if Path(name).name!=name:raise ValueError('Checkpoint pointer escaped run directory')
        restore_bc(ckpt/name,bc,sampler,identity)
    else:save_bc(ckpt,bc,sampler,identity)
    if bc.bc_updates>updates:raise ValueError('BC target budget is below restored update count')
    log=root/('metrics_attempt_'+uuid.uuid4().hex+'.jsonl')
    with log.open('x') as stream:
        while bc.bc_updates<updates:
            saved_sampler=sampler.save();indices=sampler.next(batch_size);x,a=cache.batch(indices)
            old_count=bc.bc_updates
            try:metrics=bc.update(x,a['mask'],a['label'])
            except BaseException:
                if bc.bc_updates==old_count:sampler.load(saved_sampler)
                raise
            stream.write(json.dumps(dict(metrics,sampler_epoch=sampler.epoch),allow_nan=False)+'\n');stream.flush()
            if bc.bc_updates%save_every==0 or bc.bc_updates==updates:
                save_bc(ckpt,bc,sampler,identity)
            print(f"BC {bc.bc_updates}: loss={metrics['loss']:.6g}, ADD recall={metrics['recall']}, {metrics['status']}",flush=True)
    audit=bc.freeze_audit();scores=evaluate_classifier(bc,cache)
    result=signed(dict(status='CPU_DIAGNOSTIC_COMPLETE' if source['diagnostic'] else 'BC_COMPLETE',
        identity=identity,bc_updates=bc.bc_updates,freeze_audit=audit,classification=scores,
        original_joint_optimizer_updated=False,formal_gpu_training=device!='cpu',dev_evaluated=False,
        log_file=log.name,checkpoint=str(ckpt/(ckpt/'latest').read_text().strip())))
    write_json(root/'training_receipt.json',result)
    return result
