#!/usr/bin/env python3
"""Shared source audit and actual transfer; CPU default, no training."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('freeze-source')
    freeze.add_argument('--source-checkpoint',required=True)
    freeze.add_argument('--output',required=True)
    for name in ('source-audit','transfer-smoke'):
        p=sub.add_parser(name)
        p.add_argument('--mode',choices=('direct_init','direct_full','after_bc_s3a'),required=True)
        p.add_argument('--catalog',required=True)
        p.add_argument('--source-checkpoint')
        p.add_argument('--base-checkpoint')
        p.add_argument('--features')
        p.add_argument('--config')
        p.add_argument('--seed',type=int)
        p.add_argument('--output',required=name=='transfer-smoke')
    args=parser.parse_args()
    os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root.parent/'gym-trading-env'/'src')]
    from dreamerv3.configuration import read_config_yaml
    from gym_trading_env.research.fg_s3_teacher_catalog import load_catalog,guard_output
    from gym_trading_env.research.fg_replay import write_json
    from embodied.run.fg_s3_source_admission import inspect_source,prepare_transfer,freeze_source
    from embodied.run.fg_s3_source_transfer import make_agent,apply_and_audit_transfer
    if args.command=='freeze-source':
        print(json.dumps(freeze_source(args.source_checkpoint,args.output),sort_keys=True))
        return
    cat=load_catalog(args.catalog)
    cfg=read_config_yaml(args.config) if args.config else None
    spec=inspect_source(args.mode,catalog=cat,path=args.source_checkpoint,config=cfg,
                        seed=args.seed,base_checkpoint=args.base_checkpoint,features=args.features)
    if args.command=='source-audit':
        prepare_transfer(spec)
        result=spec
    else:
        protected=[cat.root,Path(cat.manifest['source']['dataset']),Path(cat.manifest['source']['oracle_receipt']),Path(cat.manifest['labels'])]
        if args.source_checkpoint:protected.append(Path(args.source_checkpoint).resolve().parent.parent)
        if args.base_checkpoint:protected.append(Path(args.base_checkpoint).resolve().parent.parent)
        out=guard_output(args.output,protected);out.mkdir(parents=True)
        agent=make_agent(spec['origin']['config'],out)
        result=apply_and_audit_transfer(agent,spec)
        write_json(out/'source_admission_receipt.json',spec)
        write_json(out/'transfer_receipt.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
