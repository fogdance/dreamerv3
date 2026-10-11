#!/usr/bin/env python3
"""Independent S3-A features/train/audit/eval; default CPU, no joint training."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['features','train','audit','eval','smoke']:
        p=sub.add_parser(name)
        p.add_argument('--labels',required=name!='audit')
        p.add_argument('--source-checkpoint',required=name not in ['smoke','audit'])
        p.add_argument('--output',required=name!='audit')
        p.add_argument('--seed',type=int,default=0);p.add_argument('--run-id',default='S3A')
        p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
        p.add_argument('--max-cpu-workers',type=int,choices=[1],default=1)
        p.add_argument('--diagnostic',action='store_true');p.add_argument('--dry-run',action='store_true')
        p.add_argument('--dataset');p.add_argument('--oracle-receipt');p.add_argument('--candidate',choices=['C1'],default='C1')
        if name in ['train','audit']:p.add_argument('--features',required=name=='train')
        if name in ['train','smoke']:
            p.add_argument('--updates',type=int,default=10);p.add_argument('--batch-size',type=int,default=256)
            p.add_argument('--learning-rate',type=float,default=4e-5)
            p.add_argument('--resume',action='store_true');p.add_argument('--save-every',type=int,default=100)
        if name in ['features','smoke']:
            p.add_argument('--max-rows',type=int);p.add_argument('--shard-rows',type=int,default=1024)
        if name=='smoke':p.add_argument('--compute-dtype',choices=['float32','bfloat16'],default='float32')
        if name in ['audit','eval']:p.add_argument('--checkpoint')
        if name=='audit':p.add_argument('--run')
        if name=='eval':
            p.add_argument('--prepared',required=True)
            p.add_argument('--strategy',choices=['original','bc','always_wait','always_add','random_add','delay'],default='bc')
    args=parser.parse_args(argv)
    if args.command=='smoke' and args.device!='cpu':parser.error('smoke is CPU-only')
    if args.device=='cpu':os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from gym_trading_env.research.fg_s3a_labels import load_labels
    from embodied.run.fg_s3a_checkpoint import validate_source,load_source,diagnostic_source,audit_bc
    from embodied.run.fg_s3a_features import build_features,FeatureCache
    from embodied.run.fg_s3a_train import train
    if args.command=='audit':
        if args.checkpoint:result=audit_bc(args.checkpoint)
        elif args.run:
            from gym_trading_env.research.fg_replay import read_manifest
            result=read_manifest(Path(args.run)/'training_receipt.json')
            audit_bc(result['checkpoint'],result['identity'])
        else:parser.error('audit requires --checkpoint or --run')
    else:
        labels=load_labels(args.labels)
        if args.dataset and Path(args.dataset).resolve()!=Path(labels['source']['dataset']):parser.error('dataset differs from labels')
        if args.oracle_receipt and Path(args.oracle_receipt).resolve()!=Path(labels['source']['oracle_receipt']):parser.error('Oracle receipt differs from labels')
        if args.command=='smoke':
            if args.resume:parser.error('Resume an existing smoke with train --diagnostic --resume')
            if args.dry_run:
                print(json.dumps(dict(source='NEW_RANDOM_SMALL_ORIGINAL_AGENT',device='cpu',labels=labels['summary'])));return
            root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
            agent,source=diagnostic_source(labels,root/'source',compute_dtype=args.compute_dtype)
            features=build_features(agent,source,args.labels,root/'features',seed=args.seed,
                run_id=args.run_id,shard_rows=args.shard_rows,max_rows=1100 if args.max_rows is None else args.max_rows)
            cache=FeatureCache(root/'features',source,args.labels)
            result=train(source,cache,root/'train',seed=args.seed,run_id=args.run_id,
                batch_size=args.batch_size,updates=args.updates,opt_config=dict(lr=args.learning_rate),
                save_every=args.save_every,agent=agent)
        else:
            source=validate_source(args.source_checkpoint,labels,diagnostic=args.diagnostic)
            if args.dry_run:
                print(json.dumps(source,indent=2));return
            if args.command=='features':
                agent=load_source(source,args.output,device=args.device)
                result=build_features(agent,source,args.labels,args.output,seed=args.seed,
                    run_id=args.run_id,shard_rows=args.shard_rows,max_rows=args.max_rows)
            elif args.command=='train':
                cache=FeatureCache(args.features,source,args.labels)
                result=train(source,cache,args.output,seed=args.seed,run_id=args.run_id,
                    batch_size=args.batch_size,updates=args.updates,opt_config=dict(lr=args.learning_rate),
                    resume=args.resume,save_every=args.save_every,device=args.device)
            else:
                from embodied.run.fg_s3a_eval import evaluate
                result=evaluate(source,args.labels,args.prepared,args.output,args.strategy,
                                checkpoint=args.checkpoint,device=args.device)
    print(json.dumps({k:v for k,v in result.items() if k not in ['freeze_audit','source','files','source_param_digests']},indent=2))


if __name__=='__main__':main()
