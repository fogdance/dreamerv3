"""Actor-only FULL/MATCHED/CERTIFIED loop over exactly S3-A's causal features."""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from dreamerv3.fg_s3a_actor import DEFAULT_OPT
from dreamerv3.fg_s3b_bc import WeightedActorBC
from embodied.run.fg_s3a_checkpoint import load_source,package_versions
from embodied.run.fg_s3a_train import Sampler,evaluate_classifier
from embodied.run.fg_s3b_checkpoint import PROTOCOL,code_identity,restore_bc,save_bc
from embodied.run.fg_checkpoint import file_hash
from gym_trading_env.research.fg_replay import read_manifest,signed,write_json
from gym_trading_env.research.fg_s3b_weights import MODES,bind_feature_cache


class TrainingRows:
    def __init__(self,cache,weights,mode,*,diagnostic=False):
        if mode not in MODES: raise ValueError('Unknown S3-B comparison mode')
        self.manifest,self.raw_weights = bind_feature_cache(weights,cache,diagnostic=diagnostic)
        if (mode != 'S3A_FULL' and not diagnostic
                and self.manifest['summary']['training_admission'] != 'PASS'):
            raise ValueError(self.manifest['summary']['training_admission'])
        self.mode,self.cache = mode,cache
        self.indices = (np.arange(cache.rows,dtype=np.int64) if mode == 'S3A_FULL'
                        else np.flatnonzero(self.raw_weights > 0).astype(np.int64))
        if not len(self.indices): raise ValueError('INSUFFICIENT_SUPERVISION_COVERAGE: empty feature subset')
        self.index_hash = hashlib.sha256(self.indices.tobytes()).hexdigest()

    def batch(self,local_indices):
        local_indices = np.asarray(local_indices)
        if (local_indices.ndim != 1 or not np.issubdtype(local_indices.dtype,np.integer)
                or (local_indices < 0).any() or (local_indices >= len(self.indices)).any()):
            raise ValueError('Invalid comparison subset indices')
        global_indices = self.indices[local_indices]
        x,a = self.cache.batch(global_indices)
        w = (self.raw_weights[global_indices] if self.mode == 'S3B_CERTIFIED'
             else np.ones(len(global_indices),np.float64))
        return x,a,w,global_indices


def training_identity(source,cache,rows,weights,*,seed,run_id,batch_size,opt_config,
                      comparison_updates=1000,config_sha256=None):
    return dict(protocol=PROTOCOL,mode=rows.mode,source_manifest_sha256=source['manifest_sha256'],
        source_identity_hash=source['identity_hash'],diagnostic=source['diagnostic'],
        feature_manifest_sha256=file_hash(cache.root/'manifest.json'),
        label_manifest_sha256=cache.manifest['labels_manifest_sha256'],
        values_manifest_sha256=rows.manifest['values_manifest_sha256'],
        weights_manifest_sha256=file_hash(Path(weights)/'manifest.json'),
        scale_sha256=rows.manifest['scale_sha256'],matched_index_sha256=rows.manifest['matched_index_sha256'],
        feature_subset_sha256=rows.index_hash,feature_subset_rows=len(rows.indices),
        seed=seed,run_id=run_id,batch_size=batch_size,comparison_updates=comparison_updates,
        optimizer=dict(DEFAULT_OPT,**(opt_config or {})),sampler='uniform_permutation_same_matched_indices',
        code_sha256=code_identity(),packages=package_versions(),config_sha256=config_sha256)


def train(source,cache,weights,output,*,mode='S3B_CERTIFIED',seed=0,run_id='s3b',batch_size=256,
          updates=1000,comparison_updates=1000,opt_config=None,resume=False,save_every=100,
          device='cpu',agent=None,config_sha256=None):
    if updates < 1 or batch_size < 1 or save_every < 1 or comparison_updates < updates:
        raise ValueError('Invalid S3-B comparison/update budget')
    rows = TrainingRows(cache,weights,mode,diagnostic=source['diagnostic'])
    root = Path(output).resolve()
    # Protect the entire original source experiment, including logs and every
    # checkpoint, rather than just the selected checkpoint leaf directory.
    inputs = [Path(source['path']).resolve().parent.parent,cache.root,Path(weights).resolve(),
              Path(rows.manifest['labels']),Path(rows.manifest['values'])]
    vm = read_manifest(Path(rows.manifest['values'])/'manifest.json')
    inputs += [Path(vm['identity'][name]) for name in ('dataset','oracle_receipt')]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in inputs):
        raise ValueError('S3-B output overlaps immutable source/cache/checkpoint')
    identity = training_identity(source,cache,rows,weights,seed=seed,run_id=run_id,
        batch_size=batch_size,opt_config=opt_config,comparison_updates=comparison_updates,config_sha256=config_sha256)
    if resume:
        if read_manifest(root/'run_identity.json')['identity'] != identity:
            raise ValueError('S3-B resume experiment/source/weights/feature identity differs')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/'run_identity.json',signed(dict(identity=identity)))
    agent = agent or load_source(source,root,device=device)
    bc = WeightedActorBC(agent,opt_config)
    if bc.source_digests != cache.manifest['source_param_digests']:
        raise ValueError('Loaded source differs from feature-producing model')
    sampler = Sampler(len(rows.indices),seed)
    ckpt = root/'ckpt'
    if resume:
        name = (ckpt/'latest').read_text().strip()
        if Path(name).name != name: raise ValueError('S3-B checkpoint pointer escaped run directory')
        restore_bc(ckpt/name,bc,sampler,identity)
    else:
        save_bc(ckpt,bc,sampler,identity)
    if bc.bc_updates > updates: raise ValueError('Target is below restored update count')
    log = root/('metrics_attempt_'+uuid.uuid4().hex+'.jsonl')
    with log.open('x') as f:
        while bc.bc_updates < updates:
            old_sampler = sampler.save()
            indices = sampler.next(batch_size)
            x,a,w,global_indices = rows.batch(indices)
            old_count = bc.bc_updates
            try:
                metrics = bc.update(x,a['mask'],a['label'],w)
            except BaseException:
                if old_count == bc.bc_updates: sampler.load(old_sampler)
                raise
            # Selected CERTIFIED rows are strictly positive; a zero batch here
            # indicates data corruption rather than a reason to loop forever.
            if metrics['skipped']: raise RuntimeError('Matched interface produced an impossible zero batch')
            metrics.update(mode=mode,sampler_epoch=sampler.epoch,
                batch_index_sha256=hashlib.sha256(global_indices.tobytes()).hexdigest(),
                batch_weight_sha256=hashlib.sha256(w.tobytes()).hexdigest())
            f.write(json.dumps(metrics,allow_nan=False)+'\n'); f.flush()
            if bc.bc_updates % save_every == 0 or bc.bc_updates == updates:
                save_bc(ckpt,bc,sampler,identity)
            print(f"{mode} {bc.bc_updates}: loss={metrics['loss']:.6g}, ADD recall={metrics['recall']}, {metrics['status']}",flush=True)
    result = signed(dict(status='CPU_DIAGNOSTIC_COMPLETE' if source['diagnostic'] else 'BC_COMPLETE',
        identity=identity,bc_updates=bc.bc_updates,freeze_audit=bc.freeze_audit(),
        classification=evaluate_classifier(bc,cache),target_updates=updates,
        original_joint_optimizer_updated=False,formal_gpu_training=device != 'cpu',dev_evaluated=False,
        checkpoint=str(ckpt/(ckpt/'latest').read_text().strip()),log_file=log.name,
        interpretation='TRAIN imitation; FULL vs MATCHED includes selection; MATCHED vs CERTIFIED isolates weighting'))
    write_json(root/'training_receipt.json',result)
    return result
