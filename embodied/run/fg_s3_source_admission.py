"""Read-only identity admission for new experiments, separate from exact resume."""
from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import json
import pickle
import subprocess

import numpy as np

from embodied.run.fg_checkpoint import file_hash, plain, validate_bundle, learning_spec
from embodied.run.fg_s3a_checkpoint import audit_bc, package_versions
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, signed

PROTOCOL = 'FG_S3_SOURCE_ADMISSION_V1'


def _normalized_code(data, name):
    tree = ast.parse(data)
    for item in tree.body:
        if isinstance(item, ast.ClassDef):
            if name == 'dreamerv3/agent.py' and item.name == 'Agent':
                item.body = [x for x in item.body if not isinstance(x, ast.FunctionDef)
                             or x.name != 'training_loss_fn']
                for fn in item.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == 'train':
                        for call in ast.walk(fn):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'opt':
                                call.args[0] = ast.Attribute(ast.Name('self', ast.Load()), 'loss', ast.Load())
            if name == 'embodied/jax/outs.py' and item.name == 'MaskedCategorical':
                item.body = [x for x in item.body if not isinstance(x, ast.FunctionDef) or x.name != 'prob']
    return ast.dump(tree, include_attributes=False)


@lru_cache(maxsize=64)
def _verify_code(name, expected, actual):
    root = Path(__file__).resolve().parents[2]
    if name.startswith('gym_trading_env/'):
        repo, rel = root.parent/'gym-trading-env', 'src/'+name
    else:
        repo, rel = root, name
    path = repo/rel
    if file_hash(path) == expected:
        return dict(source_sha256=expected, target_sha256=actual, disposition='IDENTICAL')
    # These are explicit transfer changes, not an old-run resume bypass:
    # the generic loss factory and already reviewed safe probability diagnostic.
    if name not in ('dreamerv3/agent.py', 'embodied/jax/outs.py'):
        raise ValueError('Unsupported source code transition: '+name)
    revisions = subprocess.check_output(['git', 'rev-list', '--all', '--', rel], cwd=repo, text=True).split()
    for rev in revisions:
        data = subprocess.check_output(['git', 'show', rev+':'+rel], cwd=repo)
        if hashlib.sha256(data).hexdigest() == expected:
            if _normalized_code(data, name) != _normalized_code(path.read_bytes(), name):
                raise ValueError('Source network/training semantics changed: '+name)
            return dict(source_sha256=expected, target_sha256=actual,
                        disposition='AUDITED_TRANSFER_ONLY', source_git=rev)
    raise ValueError('Source producer bytes unavailable: '+name)


def verify_producer_code(mapping):
    root = Path(__file__).resolve().parents[2]
    result = {}
    for name, expected in mapping.items():
        path = root.parent/'gym-trading-env'/'src'/name if name.startswith('gym_trading_env/') else root/name
        result[name] = _verify_code(name, expected, file_hash(path))
    return result


def _match_training(identity, catalog):
    m = catalog.manifest
    expected = dict(candidate='C1', pool='C1_natural_oracle',
                    dataset_sha256=m['source']['dataset_sha256'],
                    reward_params_hash=m['source']['reward_params_hash'],
                    observation_identity=m['observation_identity'])
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('Source/catalog TRAIN/Reward/Observation identity mismatch')


def _full(path, catalog):
    path = Path(path).resolve()
    meta = validate_bundle(path)
    parent = path.parent.parent
    diagnostic = (parent/'s3a_diagnostic_identity.json').exists()
    frozen = False
    if (parent/'source_snapshot.json').exists():
        snapshot = read_manifest(parent/'source_snapshot.json')
        frozen = snapshot['manifest_sha256'] == file_hash(path/'manifest.json')
    if not diagnostic and not frozen:
        summary_path = parent/'training_summary.json'
        if not summary_path.exists():
            raise ValueError('SOURCE_REQUIRES_FROZEN_SNAPSHOT_OR_FINAL_SUMMARY')
        summary = json.loads(summary_path.read_text())
        frozen = (summary.get('status') == 'TRAINING_COMPLETE'
                  and summary.get('last_update') == meta['step']
                  and summary.get('checkpoint_manifest_sha256') == file_hash(path/'manifest.json'))
    if not diagnostic and not frozen:
        raise ValueError('SOURCE_REQUIRES_FROZEN_SNAPSHOT_OR_FINAL_SUMMARY')
    tag = read_manifest(parent/'s3a_diagnostic_identity.json') if diagnostic else json.loads(
        (parent/'single3_identity.json').read_text())
    identity = tag['identity']
    if fingerprint(identity) != tag['identity_hash'] or meta['identity']['identity_hash'] != tag['identity_hash']:
        raise ValueError('Source identity signature mismatch')
    if meta['identity']['protocol'] not in ('FG_SINGLE3_OFFLINE_V1', 'FG_S3A_DIAGNOSTIC_SOURCE_V1'):
        raise ValueError('Unsupported base source identity protocol')
    _match_training(identity, catalog)
    from dreamerv3.configuration import read_config_yaml
    config = read_config_yaml(path/'config.yaml')
    if (config['agent'] != identity['agent_config'] or plain(learning_spec(config)) != identity['learning']
            or config['agent']['horizon'] != 333 or not config['agent']['contdisc']):
        raise ValueError('Source model/learning config differs from signed identity')
    packages = package_versions()
    if any(packages.get(k) != v for k, v in identity['packages'].items()):
        raise ValueError('Source dependency version mismatch')
    code = verify_producer_code(identity['code_sha256'])
    return dict(path=str(path), manifest_sha256=file_hash(path/'manifest.json'),
                identity=identity, identity_hash=tag['identity_hash'], config=config,
                step=meta['step'], diagnostic=diagnostic, producer_code=code)


def inspect_source(kind, *, catalog, path=None, seed=None, config=None,
                   base_checkpoint=None, features=None):
    if kind == 'resume_s3c':
        raise ValueError('Resume must use the same run FullCheckpoint, not source transfer')
    if kind not in ('direct_init', 'direct_full', 'after_bc_s3a'):
        raise ValueError('NOT_SUPPORTED: '+str(kind))
    if kind == 'direct_init':
        if path is not None or seed is None or config is None:
            raise ValueError('direct_init requires explicit seed and config, no checkpoint')
        cfg = plain(config)
        if cfg['seed'] != seed or cfg['agent']['horizon'] != 333 or not cfg['agent']['contdisc']:
            raise ValueError('Initialization seed/Reward horizon differs')
        origin = dict(config=cfg, step=0, diagnostic=bool(catalog.manifest['diagnostic']))
    elif kind == 'direct_full':
        if path is None:
            raise ValueError('direct_full requires a source checkpoint')
        origin = _full(path, catalog)
    else:
        if path is None:
            raise ValueError('after_bc_s3a requires an atomic S3-A package')
        path = Path(path).resolve()
        audited = audit_bc(path)
        state = pickle.loads((path/'state.pkl').read_bytes())
        identity = audited['identity']
        if identity['label_manifest_sha256'] != catalog.manifest['labels_manifest_sha256']:
            raise ValueError('BC/catalog label identity mismatch')
        fm = None
        if features:
            fm = read_manifest(Path(features)/'manifest.json')
            if (file_hash(Path(features)/'manifest.json') != identity['feature_manifest_sha256']
                    or fm['source']['manifest_sha256'] != identity['source_manifest_sha256']
                    or fm['source']['identity_hash'] != identity['source_identity_hash']):
                raise ValueError('BC feature/source provenance mismatch')
            if base_checkpoint is None and Path(fm['source']['path']).is_dir():
                base_checkpoint = fm['source']['path']
        if base_checkpoint:
            base = _full(base_checkpoint, catalog)
            if base['manifest_sha256'] != identity['source_manifest_sha256'] or base['identity_hash'] != identity['source_identity_hash']:
                raise ValueError('BC baseline reference differs')
            cfg = base['config']
        else:
            if fm is None:
                raise ValueError('BC transfer needs hash-bound baseline or feature provenance')
            producer = fm['source']
            if fingerprint(producer['identity']) != identity['source_identity_hash']:
                raise ValueError('BC feature producer identity differs')
            _match_training(producer['identity'], catalog)
            verify_producer_code(producer['identity']['code_sha256'])
            cfg = producer['config']
            if cfg['agent'] != producer['identity']['agent_config']:
                raise ValueError('BC feature config differs from producer')
            base = None
        # Full Agent state and proof were audited above; independent BC opt is
        # deliberately left outside the transfer state.
        origin = dict(path=str(path), manifest_sha256=file_hash(path/'manifest.json'),
                      bc_identity=identity, bc_updates=state['bc_updates'],
                      config=cfg, step=state['agent']['counters']['updates'],
                      diagnostic=identity['diagnostic'], baseline=base,
                      baseline_optimizer_equality='PENDING' if base else 'NOT_VERIFIED')
    receipt = signed(dict(protocol=PROTOCOL, kind=kind, origin=origin,
        catalog_manifest_sha256=catalog.manifest_sha256,
        dataset_sha256=catalog.manifest['source']['dataset_sha256'],
        reward_params_hash=catalog.manifest['source']['reward_params_hash'],
        observation_identity=catalog.manifest['observation_identity'],
        optimizer_transfer_policy='retain_base_joint_opt' if kind == 'after_bc_s3a' else 'original_direct',
        code_sha256=file_hash(__file__), packages=package_versions()))
    return receipt


def prepare_transfer(spec, target_schema=None):
    if spec['protocol'] != PROTOCOL or fingerprint({k:v for k,v in spec.items() if k!='content_hash'}) != spec['content_hash']:
        raise ValueError('Source admission receipt corrupt')
    origin = spec['origin']
    if spec['kind'] == 'direct_init':
        return None
    path = Path(origin['path'])
    if file_hash(path/'manifest.json') != origin['manifest_sha256']:
        raise ValueError('Source changed after admission')
    if spec['kind'] == 'direct_full':
        validate_bundle(path)
        state = pickle.loads((path/'agent.pkl').read_bytes())
    else:
        audit_bc(path)
        state = pickle.loads((path/'state.pkl').read_bytes())['agent']
    required = {'params','counters','policy_params','pending_sync','pending_outs','pending_mets'}
    if set(state) != required or state['counters']['updates'] != origin['step']:
        raise ValueError('Incomplete full Agent state/counters')
    if any('s3a_bc_opt' in k or 's3b_bc_opt' in k for k in state['params']):
        raise ValueError('Independent BC optimizer in Agent state')
    for k, v in state['params'].items():
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite source parameter: '+k)
    if target_schema is not None:
        actual = {k:(v.shape,str(v.dtype)) for k,v in state['params'].items()}
        if actual != target_schema:
            raise ValueError('Source parameter keys/dtypes/shapes differ from target')
    return state


def freeze_source(path, output):
    """Copy one published node and its identity, hash before/after, never hardlink."""
    import shutil
    import os
    from gym_trading_env.research.fg_s3_teacher_catalog import guard_output
    from gym_trading_env.research.fg_replay import write_json
    source = Path(path).resolve()
    validate_bundle(source)
    initial = file_hash(source/'manifest.json')
    root = guard_output(output, (source.parent.parent,))
    root.mkdir(parents=True)
    dest = root/'ckpt'/source.name
    shutil.copytree(source, dest)
    tags = [name for name in ('single3_identity.json','s3a_diagnostic_identity.json')
            if (source.parent.parent/name).is_file()]
    if len(tags) != 1:
        raise ValueError('Source snapshot requires one parent identity')
    shutil.copyfile(source.parent.parent/tags[0],root/tags[0])
    validate_bundle(dest)
    validate_bundle(source)
    if file_hash(source/'manifest.json') != initial or file_hash(dest/'manifest.json') != initial:
        raise ValueError('Source changed during snapshot copy')
    receipt = signed(dict(protocol=PROTOCOL, source=str(source), snapshot=str(dest),
        manifest_sha256=initial, parent_identity_sha256=file_hash(root/tags[0])))
    write_json(root/'source_snapshot.json',receipt)
    for p in root.rglob('*'):
        if p.is_file(): os.chmod(p,0o444)
    return receipt


def transfer_control_state(spec):
    """Public transfer boundary for external Gate/window state; never restore RNG/carry."""
    origin = spec['origin']
    base = origin.get('baseline') if spec['kind']=='after_bc_s3a' else origin
    if spec['kind']=='direct_init':
        return dict(complete=True, new_experiment=True, gate=None, aggregate=None, samples=0,
                    opening=None)
    if not base or not base.get('path'):
        return dict(complete=not bool(origin['config']['action_mask_warmup']['enabled']),
                    new_experiment=True, gate=None, aggregate=None, samples=0,
                    opening=None)
    path=Path(base['path'])
    if file_hash(path/'manifest.json')!=base['manifest_sha256']:
        raise ValueError('Control-state baseline changed')
    validate_bundle(path)
    enabled=bool(base['config']['action_mask_warmup']['enabled'])
    gate_file, runtime_file = path/'action_mask_warmup.pkl', path/'runtime.pkl'
    gate=pickle.loads(gate_file.read_bytes()) if gate_file.is_file() else None
    runtime=pickle.loads(runtime_file.read_bytes()) if runtime_file.is_file() else {}
    aggregate=runtime.get('aggregates',{}).get('warmup')
    return dict(complete=not enabled or (gate is not None and aggregate is not None),
                new_experiment=True,gate=gate,aggregate=aggregate,samples=runtime.get('samples',0),
                opening=base.get('identity',{}).get('mask_gate_opening'))
