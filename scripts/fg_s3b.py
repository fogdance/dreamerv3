"""Distinct S3-B entry; default CPU, no call to the original joint train()."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('train','admission-audit'):
        p = sub.add_parser(name)
        for key in ('config','dataset','oracle-receipt','s3a-label-index','weights','feature-cache',
                    'source-checkpoint','output','run-id'):
            p.add_argument('--'+key)
        p.add_argument('--mode',choices=['S3A_FULL','S3A_MATCHED','S3B_CERTIFIED'])
        p.add_argument('--seed',type=int)
        p.add_argument('--batch-size',type=int)
        p.add_argument('--updates',type=int)
        p.add_argument('--comparison-updates',type=int)
        p.add_argument('--save-every',type=int)
        p.add_argument('--resume',action='store_true')
        p.add_argument('--diagnostic',action='store_true')
        p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    p = sub.add_parser('checkpoint-audit'); p.add_argument('--checkpoint',required=True)
    args = parser.parse_args()
    # Set the platform BEFORE importing any Embodied/JAX code.
    device = getattr(args,'device','cpu')
    os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
    if device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    from embodied.run.fg_s3b_checkpoint import audit_bc
    if args.command == 'checkpoint-audit':
        m = audit_bc(args.checkpoint)
        print(json.dumps(dict(status='PASS',protocol=m['protocol'],bc_updates=m['bc_updates'],
                             frozen_keys=m['freeze_audit']['frozen_keys'],mode=m['identity']['mode']),indent=2))
        return
    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        if cfg.get('schema') != 'FG_S3B_TRAIN_CONFIG_V1' or set(cfg)-{
            'schema','dataset','oracle_receipt','s3a_label_index','weights','feature_cache','source_checkpoint',
            'mode','seed','batch_size','updates','comparison_updates','save_every','optimizer'}:
            raise ValueError('Foreign/unknown S3-B train configuration')
    def field(name,default=None):
        value = getattr(args,name,None)
        if value is None: value = cfg.get(name,default)
        if value is None: parser.error('Missing --'+name.replace('_','-'))
        return value
    from gym_trading_env.research.fg_s3a_labels import load_labels
    from gym_trading_env.research.fg_s3b_weights import audit_weights
    from embodied.run.fg_s3a_checkpoint import validate_source
    from embodied.run.fg_s3a_features import FeatureCache
    from embodied.run.fg_s3b_train import TrainingRows,train
    from embodied.run.fg_checkpoint import file_hash
    labels = field('s3a_label_index')
    lm = load_labels(labels)
    wm = audit_weights(field('weights'))
    from gym_trading_env.research.fg_replay import read_manifest
    vm = read_manifest(Path(wm['values'])/'manifest.json')
    for name,key in [('dataset','dataset'),('oracle_receipt','oracle_receipt'),('s3a_label_index','s3a_labels')]:
        supplied = getattr(args,name,None) or cfg.get(name)
        if supplied and Path(supplied).resolve() != Path(vm['identity'][key]).resolve():
            raise ValueError('Requested source differs from immutable weights: '+name)
    try:
        source = validate_source(field('source_checkpoint'),lm,diagnostic=args.diagnostic)
    except (ValueError,FileNotFoundError,KeyError) as error:
        if args.command != 'admission-audit': raise
        print(json.dumps(dict(status='WAITING_FOR_SOURCE_CHECKPOINT',reason=str(error),
            data_proof_status='PASS',weighted_supervision_status=wm['summary']['training_admission'],
            formal_gpu_training='NOT_RUN'),ensure_ascii=False,indent=2))
        return
    cache = FeatureCache(field('feature_cache'),source,labels)
    if args.command == 'admission-audit':
        rows = TrainingRows(cache,field('weights'),field('mode','S3B_CERTIFIED'),diagnostic=source['diagnostic'])
        print(json.dumps(dict(status='PASS',source_manifest_sha256=source['manifest_sha256'],
                             feature_subset_rows=len(rows.indices),feature_subset_sha256=rows.index_hash),indent=2))
        return
    result = train(source,cache,field('weights'),field('output'),mode=field('mode','S3B_CERTIFIED'),
        seed=field('seed',0),run_id=field('run_id','s3b'),batch_size=field('batch_size',256),
        updates=field('updates',1000),comparison_updates=field('comparison_updates',1000),
        opt_config=cfg.get('optimizer'),resume=args.resume,save_every=field('save_every',100),device=device,
        config_sha256=file_hash(args.config) if args.config else None)
    print(json.dumps(dict(status=result['status'],bc_updates=result['bc_updates'],
        checkpoint=result['checkpoint'],classification=result['classification']['status']),indent=2))


if __name__ == '__main__':
    main()
