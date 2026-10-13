"""Original policy teacher-history features; immutable streamed float32 shards."""
from pathlib import Path
import json

import jax
import numpy as np

from embodied.run.fg_checkpoint import file_hash
from embodied.run.fg_s3a_checkpoint import code_identity,package_versions
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import (FIELDS, S3_FIELDS, check_files, iter_replay_rows,
                                               read_manifest, signed, write_json)
from gym_trading_env.research.fg_s3a_labels import INDEX_DTYPE, load_labels

PROTOCOL='FG_S3A_FEATURES_V1'


def teacher_step(agent,carry,step):
    if set(step)!=set(S3_FIELDS)|{'stepid'} or set(agent.obs_space)!=set(FIELDS)-{'action'}:
        raise ValueError('Only the original causal Replay observation schema may enter Encoder')
    obs={k:np.asarray(step[k])[None] for k in agent.obs_space}
    carry,_,out=agent.policy(carry,obs,mode='eval')
    # RSSM._observe returns exactly the same deter/stoch in feat and carry.
    # feat2tensor uses only these two fields; no duplicate RSSM implementation.
    dyn=carry[1]
    feat=np.concatenate([np.asarray(jax.device_get(dyn['deter'][0]),np.float32),
                         np.asarray(jax.device_get(dyn['stoch'][0]),np.float32).reshape(-1)])
    carry=(*carry[:3],{agent.model.action_key:[np.asarray(step['action']).copy()]})
    return carry,feat,out


def build_features(agent,source,labels,output,*,seed=0,run_id='features',shard_rows=1024,max_rows=None):
    root=Path(output).resolve();labelroot=Path(labels).resolve()
    m=load_labels(labelroot)
    if shard_rows<1 or (max_rows is not None and (max_rows<1 or not source['diagnostic'])):
        raise ValueError('Prefix extraction is permitted only for an explicit diagnostic source')
    if root.is_relative_to(Path(source['path'])) or root.is_relative_to(labelroot) or root.is_relative_to(Path(m['source']['dataset'])):
        raise ValueError('Features output cannot be inside an immutable input')
    root.mkdir(parents=True,exist_ok=False)
    before=parameter_digests(agent.params)
    with agent.n_actions.lock:agent.n_actions.value=0
    # Wrapper _seeds derives each key from [config.seed, n_actions].
    # Change only the feature process's seed; model architecture/weights stay exact.
    old_seed=agent.config.seed;agent.config=agent.config.update(seed=seed)
    shards=[];values=[];indexes=[];visited=0;current_contract=None;max_error=0.

    def flush():
        if not values:return
        n=len(shards);prefix=f'shard_{n:06d}'
        x=np.stack(values).astype(np.float32);a=np.array(indexes,dtype=INDEX_DTYPE)
        np.save(root/(prefix+'.features.npy'),x,allow_pickle=False)
        np.save(root/(prefix+'.index.npy'),a,allow_pickle=False)
        shards.append(dict(features=prefix+'.features.npy',index=prefix+'.index.npy',
                           contract=current_contract,rows=len(x),width=x.shape[1]))
        values.clear();indexes.clear()

    try:
        for rec in m['traces']:
            if max_rows is not None and visited>=max_rows:break
            current_contract=rec['contract']
            index=np.load(labelroot/rec['labels_file'],mmap_mode='r',allow_pickle=False)
            pos=0;carry=agent.init_policy(1)
            trace=Path(m['source']['dataset'])/rec['directory'];tm=read_manifest(trace/'manifest.json')
            for ordinal,step in iter_replay_rows(trace,tm):
                if max_rows is not None and visited>=max_rows:break
                carry,x,out=teacher_step(agent,carry,step);visited+=1
                if pos<len(index) and ordinal==int(index[pos]['ordinal']):
                    row=index[pos]
                    if (not np.array_equal(row['stepid'],step['stepid']) or row['label']!=step['action']
                            or not np.array_equal(row['mask'],step['action_mask'])):
                        raise ValueError('Feature/label join identity mismatch')
                    if not np.isfinite(x).all():raise FloatingPointError('Nonfinite cached feature')
                    if 'p_add' in out:
                        # Verify storage promotion preserves the actual policy probability.
                        import ninjax as nj
                        def probability(value,mask):
                            dist=agent.model._masked_policy(agent.model.pol(value,1),mask,False)[agent.model.action_key]
                            return jax.nn.softmax(dist.logits)[...,1]
                        if not hasattr(agent,'_s3a_prob'):
                            f=nj.pure(probability)
                            agent._s3a_prob=jax.jit(lambda p,v,mask:f(p,v,mask,seed=0,modify=False)[1])
                        params={k:agent.params[k] for k in agent.policy_keys if k.startswith('pol/')}
                        error=float(np.max(np.abs(np.asarray(agent._s3a_prob(params,jax.device_put(x[None]),
                            jax.device_put(row['mask'][None])))-out['p_add'])))
                        max_error=max(max_error,error)
                        if error>1e-6:raise RuntimeError('Cached features change original Actor probabilities')
                    values.append(x);indexes.append(row.copy());pos+=1
                    if len(values)>=shard_rows:flush()
            flush()
            if (max_rows is None or visited<max_rows) and pos!=len(index):
                raise ValueError('Full feature extraction did not consume all trace labels')
            print(f'features {current_contract}: {pos} labels, total policy rows {visited}',flush=True)
    finally:
        agent.config=agent.config.update(seed=old_seed)
    if not shards:raise ValueError('Feature cache has no eligible labels')
    if parameter_digests(agent.params)!=before:
        raise RuntimeError('Original model state changed during feature extraction')
    total_rows=sum(t['records'] for t in m['traces'])
    if visited==total_rows and sum(s['rows'] for s in shards)!=m['summary']['counts']['eligible']:
        raise ValueError('Complete feature count differs from eligible label count')
    result=signed(dict(protocol=PROTOCOL,status='PASS',source=source,
        labels=str(labelroot),labels_manifest_sha256=file_hash(labelroot/'manifest.json'),
        seed=seed,run_id=run_id,rng='original wrapper default_rng([feature_seed, global_row_counter])',
        traversal='contract_sorted; native full ordinal order; actual policy is_first',
        feature_definition='original feat2tensor(dyn carry deter,stoch)',dtype='float32',
        shards=shards,rows=sum(s['rows'] for s in shards),visited_rows=visited,
        complete=visited==total_rows,max_original_probability_error=max_error,
        source_param_digests=before,world_model_hash=fingerprint({k:v for k,v in before.items() if k.startswith(('enc/','dyn/'))}),
        code_sha256=code_identity(),packages=package_versions(),
        files={p.name:file_hash(p) for p in root.iterdir() if p.is_file()}))
    write_json(root/'manifest.json',result)
    return result


class FeatureCache:
    def __init__(self,directory,source,labels):
        self.root=Path(directory).resolve();self.manifest=read_manifest(self.root/'manifest.json')
        m=self.manifest
        if (m['protocol']!=PROTOCOL or m['status']!='PASS' or m['dtype']!='float32'
                or m['source']['manifest_sha256']!=source['manifest_sha256']
                or m['source']['identity_hash']!=source['identity_hash']
                or m['source']['diagnostic']!=source['diagnostic']
                or m['labels_manifest_sha256']!=file_hash(Path(labels)/'manifest.json')
                or m['code_sha256']!=code_identity() or m['packages']!=package_versions()
                or (not m['complete'] and not source['diagnostic'])):
            raise ValueError('Source/labels/code/complete feature cache identity mismatch')
        load_labels(labels)
        check_files({str(self.root/p):h for p,h in m['files'].items()})
        self.features=[];self.index=[]
        for shard in m['shards']:
            x=np.load(self.root/shard['features'],mmap_mode='r',allow_pickle=False)
            a=np.load(self.root/shard['index'],mmap_mode='r',allow_pickle=False)
            if x.dtype!=np.float32 or x.shape!=(shard['rows'],shard['width']) or a.dtype!=INDEX_DTYPE or a.shape!=(len(x),):
                raise ValueError('Feature shard dtype/shape mismatch')
            self.features.append(x);self.index.append(a)
        self.ends=np.cumsum([len(x) for x in self.features]);self.rows=int(self.ends[-1])
        if self.rows!=m['rows']:raise ValueError('Feature count differs')

    def batch(self,indices):
        indices=np.asarray(indices,np.int64)
        if indices.ndim!=1 or (indices<0).any() or (indices>=self.rows).any():raise ValueError('Invalid feature row indices')
        which=np.searchsorted(self.ends,indices,side='right')
        x=np.empty((len(indices),self.features[0].shape[1]),np.float32)
        a=np.empty(len(indices),INDEX_DTYPE)
        for n in np.unique(which):
            loc=np.flatnonzero(which==n);offset=0 if n==0 else self.ends[n-1]
            x[loc]=self.features[n][indices[loc]-offset];a[loc]=self.index[n][indices[loc]-offset]
        return x,a
