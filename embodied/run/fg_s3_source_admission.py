"""Read-only identity admission for new experiments, separate from exact resume."""
from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import pickle
import subprocess

import numpy as np

from embodied.run.fg_checkpoint import file_hash, plain, validate_bundle, learning_spec, digest as state_digest, stage_origin
from embodied.run.fg_s3a_checkpoint import audit_bc, package_versions
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, signed
from dreamerv3.fg_s3_composition import (
    KEY, require_ready, validate_model_composition, validate_composition)
from gym_trading_env.research.fg_s3_reader_compat import PRODUCERS, reviewed_reader_transition

PROTOCOL = 'FG_S3_SOURCE_ADMISSION_V2'
GATE_HELPER_AST = {
    'validate_availability_threshold': 'ae187898bdf0dff46f14d0c20fc1252d11728c4c37c8fb6a354208b05308b11f',
    'availability_prediction': '3ebc25f59e6b70ec09142e935d499cdd3b6155813fdf6e4639911426e0bd9a4e',
}

def verify_dependency_versions(declared):
    """Check every producer-declared package, not only A/B's short inventory."""
    actual={}
    for name,expected in declared.items():
        try:
            actual[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError('Source dependency missing: '+name) from None
        if actual[name]!=expected:
            raise ValueError('Source dependency version mismatch: '+name)
    return actual


def runtime_versions():
    """Joint full recovery also depends on stream and snapshot serialization."""
    return {**package_versions(),**{name:importlib.metadata.version(name)
                                   for name in ('portal','cloudpickle')}}


def _normalize_gate_transfer(tree):
    """Normalize only the reviewed shared threshold delta for a new transfer."""
    helper_hashes = GATE_HELPER_AST
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in helper_hashes:
            checksum = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            if checksum != helper_hashes[node.name]:
                raise ValueError('Unreviewed availability helper transition: '+node.name)
            continue
        kept.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'hard_nonempty_mask':
            if [a.arg for a in node.args.args] == ['prob','threshold','always_available_index','raw_mask']:
                expected = ast.parse('mask = prob >= threshold if raw_mask is None else jnp.asarray(raw_mask, bool)').body[0]
                if (ast.dump(node.body[0], include_attributes=False) != ast.dump(expected, include_attributes=False)
                        or ast.dump(node.args.defaults[-1]) != ast.dump(ast.Constant(None))):
                    raise ValueError('Unreviewed mask transfer transition')
                node.args.args.pop();node.args.defaults.pop()
                node.body[0] = ast.parse('mask = prob >= threshold').body[0]
        if isinstance(node, ast.ClassDef) and node.name == 'Agent':
            validation = ast.parse('validate_availability_threshold(config.avail_threshold)').body[0]
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef):continue
                if fn.name == '__init__':
                    fn.body = [stmt for stmt in fn.body if ast.dump(stmt, include_attributes=False) != ast.dump(validation, include_attributes=False)]
                if fn.name == 'loss':
                    for stmt in fn.body:
                        if not isinstance(stmt,ast.Assign) or len(stmt.targets)!=1:continue
                        for pred,logit in (('postpred','postlogit'),('priorpred','priorlogit')):
                            expected = ast.parse(pred+' = availability_prediction('+logit+', self.config.avail_threshold)[1]').body[0]
                            if ast.dump(stmt,include_attributes=False)==ast.dump(expected,include_attributes=False):
                                stmt.value = ast.parse(logit+' >= 0',mode='eval').body
                if fn.name == '_predicted_action_mask_info':
                    expected = ast.parse('''def _predicted_action_mask_info(self, feat, bdims):
    avail = self.avail(self.feat2tensor(feat), bdims)
    prob, raw_mask = availability_prediction(self._binary_logits(avail), self.config.avail_threshold)
    return hard_nonempty_mask(prob, self.config.avail_threshold, self.config.get('avail_always_available_index', -1), raw_mask=raw_mask)
''').body[0]
                    if ast.dump(fn,include_attributes=False)==ast.dump(expected,include_attributes=False):
                        fn.body = ast.parse('''avail = self.avail(self.feat2tensor(feat), bdims)
prob = jax.nn.sigmoid(self._binary_logits(avail))
return hard_nonempty_mask(prob, self.config.avail_threshold, self.config.get('avail_always_available_index', -1))
''').body
    tree.body = kept
    return tree


def _inline_world_model(tree):
    """Accept only extraction whose expanded body is the original Agent loss.

    The helper body is not discarded or trusted by a placeholder hash: it is
    put back into loss before the complete source/target AST comparison.
    """
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != 'Agent':
            continue
        helpers = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'world_model']
        if not helpers:
            continue
        if len(helpers) != 1:
            raise ValueError('Duplicate World Model extraction')
        helper = helpers[0]
        expected = ast.parse('''def world_model(self, carry, obs, prevact, training):
    return (enc_carry, dyn_carry, dec_carry, enc_entries, dyn_entries,
            dec_entries, tokens, repfeat, reset, B, T, losses, metrics)
''').body[0]
        call = ast.parse('''(enc_carry, dyn_carry, dec_carry, enc_entries, dyn_entries,
dec_entries, tokens, repfeat, reset, B, T, losses, metrics) = self.world_model(carry, obs, prevact, training)
''').body[0]
        loss = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'loss')
        dump = lambda x: ast.dump(x, include_attributes=False)
        if (dump(helper.args) != dump(expected.args) or helper.decorator_list
                or dump(helper.body[-1]) != dump(expected.body[0])
                or dump(loss.body[0]) != dump(call)):
            raise ValueError('Unreviewed World Model extraction contract')
        loss.body = helper.body[:-1] + loss.body[1:]
        cls.body.remove(helper)
    return tree


def _normalized_code(data, name):
    tree = ast.parse(data)
    if name == 'dreamerv3/agent.py':
        tree = _inline_world_model(tree)
        tree = _normalize_gate_transfer(tree)
    if name == 'embodied/jax/internal.py':
        # Only extend the original GPU flag branch to explicit CUDA. Every
        # flag and all remaining setup code still undergo full AST comparison.
        old = ast.parse("gpuflags and platform == 'gpu'", mode='eval').body
        new = ast.parse("gpuflags and platform in ('gpu', 'cuda')", mode='eval').body
        matches = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.dump(node.test) in (ast.dump(old), ast.dump(new)):
                node.test = old
                matches += 1
        if matches != 1:
            raise ValueError('Unreviewed CUDA setup transition')
    for item in tree.body:
        if isinstance(item, ast.ClassDef):
            if name == 'dreamerv3/agent.py' and item.name == 'Agent':
                factory = ast.parse('''def training_loss_fn(self, data, effective_stepid):
    """Factory for training-only losses; report and policy stay independent."""
    return self.loss
''').body[0]
                for fn in item.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == 'training_loss_fn' and ast.dump(fn, include_attributes=False) != ast.dump(factory, include_attributes=False):
                        raise ValueError('Unreviewed training loss factory transition')
                item.body = [x for x in item.body if not isinstance(x, ast.FunctionDef)
                             or x.name != 'training_loss_fn']
                for fn in item.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == 'train':
                        for call in ast.walk(fn):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'opt':
                                approved = [ast.parse(value, mode='eval').body for value in (
                                    'self.loss', 'self.training_loss_fn(data, stepid)')]
                                if not any(ast.dump(call.args[0]) == ast.dump(value) for value in approved):
                                    raise ValueError('Unreviewed optimizer loss transition')
                                call.args[0] = ast.Attribute(ast.Name('self', ast.Load()), 'loss', ast.Load())
            if name == 'embodied/jax/outs.py' and item.name == 'MaskedCategorical':
                item.body = [x for x in item.body if not isinstance(x, ast.FunctionDef) or x.name != 'prob']
    return ast.dump(tree, include_attributes=False)


def audit_code_transition(source, target, name):
    normalized = _normalized_code(source, name)
    if normalized != _normalized_code(target, name):
        raise ValueError('Source network/training semantics changed: '+name)
    return hashlib.sha256(normalized.encode()).hexdigest()


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
    if name not in ('dreamerv3/agent.py', 'embodied/jax/outs.py', 'embodied/jax/internal.py'):
        raise ValueError('Unsupported source code transition: '+name)
    revisions = subprocess.check_output(['git', 'rev-list', '--all', '--', rel], cwd=repo, text=True).split()
    for rev in revisions:
        data = subprocess.check_output(['git', 'show', rev+':'+rel], cwd=repo)
        if hashlib.sha256(data).hexdigest() == expected:
            normalized_hash = audit_code_transition(data, path.read_bytes(), name)
            return dict(source_sha256=expected, target_sha256=actual,
                        disposition='AUDITED_TRANSFER_ONLY', source_git=rev,
                        normalized_ast_sha256=normalized_hash,
                        gate_helper_ast_sha256=GATE_HELPER_AST if name == 'dreamerv3/agent.py' else {})
    raise ValueError('Source producer bytes unavailable: '+name)


def verify_producer_code(mapping, *, warm_start=False):
    root = Path(__file__).resolve().parents[2]
    result = {}
    semantic_files = {'dreamerv3/agent.py', 'embodied/jax/opt.py', 'gym_trading_env/envs/entry_env.py',
        'gym_trading_env/rewards/single3_reward.py',
        'gym_trading_env/research/fg_reward_candidates.py'}
    for name, expected in mapping.items():
        path = root.parent/'gym-trading-env'/'src'/name if name.startswith('gym_trading_env/') else root/name
        reviewed_reader = warm_start and name in PRODUCERS and name not in semantic_files and file_hash(path) != expected
        if reviewed_reader:
            result[name] = reviewed_reader_transition(root.parent/'gym-trading-env', name, expected, file_hash)
            result[name]['disposition'] = 'AUTHENTICATED_REVIEWED_READER_PARAMETER_WARM_START'
        elif warm_start and name in semantic_files and file_hash(path) != expected:
            # A parameter warm-start deliberately changes reward/discount semantics.
            # Authenticate the actual producer bytes; apply_and_audit_transfer
            # separately requires identical NN keys, shapes, dtypes and values.
            repo = root.parent/'gym-trading-env' if name.startswith('gym_trading_env/') else root
            rel = 'src/'+name if name.startswith('gym_trading_env/') else name
            revisions = subprocess.check_output(['git','rev-list','--all','--',rel], cwd=repo, text=True).split()
            found = next((rev for rev in revisions if hashlib.sha256(
                subprocess.check_output(['git','show',rev+':'+rel],cwd=repo)).hexdigest() == expected), None)
            if found is None:
                raise ValueError('Warm-start producer bytes unavailable: '+name)
            result[name] = dict(source_sha256=expected, target_sha256=file_hash(path),
                source_git=found, disposition='AUTHENTICATED_PARAMETER_WARM_START_ONLY')
        else:
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


def _online_code(config, expected):
    """Locate producer evidence by its hash, without selecting a code version."""
    candidates = []
    for parent in Path(config['fg_entry']['manifest']).parents:
        diagnostics = parent / 'diagnostics'
        if diagnostics.is_dir():
            candidates.extend(diagnostics.rglob('*code_hashes.json'))
            break
    for path in sorted(candidates):
        mapping = json.loads(path.read_text())
        if state_digest(mapping) == expected:
            return mapping
    raise ValueError('Online producer code map unavailable for checkpoint code_hash')


def _online_full(path, meta, catalog, *, warm_start=False):
    """Admit a frozen online node; environment and model counters are distinct."""
    from gym_trading_env.research.fg_replay import check_files
    from dreamerv3.configuration import read_config_yaml
    if not (path.parent.parent / 'source_snapshot.json').is_file():
        raise ValueError('SOURCE_REQUIRES_FROZEN_ONLINE_SNAPSHOT')
    snapshot = read_manifest(path.parent.parent / 'source_snapshot.json')
    if (snapshot.get('format') != 'FG_ONLINE_FULL_SOURCE_V1'
            or snapshot['manifest_sha256'] != file_hash(path / 'manifest.json')
            or Path(snapshot['snapshot']).resolve() != path):
        raise ValueError('Online source requires an authenticated frozen snapshot')
    root = path.parent.parent
    expected_files = {'producer_code_hashes.json', 'entry_manifest.json', 'online_transfer.json',
                      'selector.yaml', 'agent_counters.json'}
    if set(snapshot['files']) != expected_files:
        raise ValueError('Incomplete online source provenance')
    check_files({str(root/n):h for n,h in snapshot['files'].items()})
    config = read_config_yaml(path / 'config.yaml')
    runtime = meta['identity']
    if (runtime.get('schema') != 1 or runtime['learning'] != plain(learning_spec(config))
            or runtime['origin'] != stage_origin(config)):
        raise ValueError('Online learning/origin identity mismatch')
    entry = read_manifest(root / 'entry_manifest.json')
    if (entry['content_hash'] != runtime['dataset']
            or entry['original_content_hash'] != catalog.manifest['source']['training_hash']
            or entry['single3_protocol'] != 'fg_single3_v1' or entry['reward_candidate'] != 'C1'
            or not entry['files'] or any(r['role'] != 'train' for r in entry['files'])):
        raise ValueError('Online source/catalog TRAIN/Reward identity mismatch')
    if (file_hash(config['fg_entry']['manifest']) != snapshot['files']['entry_manifest.json']
            or file_hash(config['run']['fg_stage_transfer']) != snapshot['files']['online_transfer.json']
            or file_hash(config['env']['gymnasium']['config_path']) != runtime['selector_hash']
            or file_hash(root / 'selector.yaml') != runtime['selector_hash']):
        raise ValueError('Online input provenance changed')
    transfer = read_manifest(root / 'online_transfer.json')
    if (transfer.get('source_kind') != 'single3_offline'
            or transfer['source_step_unit'] != 'gradient_updates'
            or transfer['target_step_unit'] != 'environment_steps'):
        raise ValueError('Unsupported online phase origin')
    base = _full(transfer['checkpoint'], catalog, warm_start=warm_start)
    if (transfer['source_identity_hash'] != base['identity_hash']
            or transfer['source_steps'] != base['step']
            or config['agent'] != base['config']['agent']
            or Path(config['fg_entry']['manifest']).resolve() != Path(base['config']['fg_entry']['manifest']).resolve()):
        raise ValueError('Online source differs from audited C1 baseline')
    mapping = json.loads((root / 'producer_code_hashes.json').read_text())
    if state_digest(mapping) != runtime['code_hash']:
        raise ValueError('Online producer code provenance mismatch')
    normalized = {n.removeprefix('dreamerv3/'):h for n,h in mapping.items()}
    code = verify_producer_code(normalized, warm_start=warm_start)
    verify_dependency_versions(runtime['versions'])
    counts = read_manifest(root / 'agent_counters.json')['counters']
    if (set(counts) != {'updates','batches','actions'}
            or any(type(v) is not int or v < 0 for v in counts.values())
            or counts['actions'] != meta['step'] or counts['updates'] < base['step']):
        raise ValueError('Online environment/model counter mismatch')
    identity = dict({k:base['identity'][k] for k in (
        'candidate','pool','dataset_sha256','reward_params_hash','observation_identity')},
        protocol='FG_SINGLE3_ONLINE_SOURCE_V1',source_kind='FROZEN_ONLINE_FG_FULL',
        model_seed=config['seed'],replay_seed=config['replay']['seed'],
        model_updates=counts['updates'],environment_steps=meta['step'],
        agent_config=config['agent'], learning=plain(learning_spec(config)),
        code_sha256=normalized, packages=runtime['versions'], runtime_provenance=runtime,
        base_manifest_sha256=base['manifest_sha256'],base_identity_hash=base['identity_hash'])
    from embodied.run.action_mask_warmup import ActionMaskWarmup
    import elements
    gate = ActionMaskWarmup.for_single3_1m(elements.Config(config['action_mask_warmup']))
    identity['mask_gate_opening'] = dict(thresholds=gate.opening_thresholds,
        consecutive_reports=gate.required_passes, automatic_activation=gate.enabled)
    return dict(path=str(path), manifest_sha256=file_hash(path/'manifest.json'), identity=identity,
        identity_hash=fingerprint(identity), config=config, step=counts['updates'],
        checkpoint_step=meta['step'], checkpoint_step_unit='environment_steps',
        environment_steps=meta['step'], agent_counters=counts, diagnostic=base['diagnostic'],
        producer_code=code, provenance_files={str(root/n):h for n,h in snapshot['files'].items()} |
        {str(root/'source_snapshot.json'):file_hash(root/'source_snapshot.json')})


def _full(path, catalog, *, warm_start=False):
    path = Path(path).resolve()
    meta = validate_bundle(path)
    if meta['identity'].get('schema') == 1 and 'code_hash' in meta['identity']:
        return _online_full(path, meta, catalog, warm_start=warm_start)
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
    from dreamerv3.configuration import read_config_yaml
    config = read_config_yaml(path/'config.yaml')
    if warm_start:
        dataset = Path(config['fg_single3_offline']['dataset'])/'manifest.json'
        old = read_manifest(dataset)
        from gym_trading_env.envs.entry_contract import read_yaml
        entry = read_manifest(config['fg_entry']['manifest'])
        reward = read_yaml(entry['direction_path'])['reward']
        if (file_hash(dataset) != identity['dataset_sha256'] or old['status'] != 'PASS'
                or old['identity']['original_training_hash'] != catalog.manifest['source']['training_hash']
                or fingerprint(reward) != identity['reward_params_hash']
                or identity['candidate'] != 'C1' or identity['pool'] != 'C1_natural_oracle'):
            raise ValueError('Warm-start original TRAIN/Reward evidence differs')
    else:
        _match_training(identity, catalog)
    if (config['agent'] != identity['agent_config'] or plain(learning_spec(config)) != identity['learning']
            or config['agent']['horizon'] != 333 or not config['agent']['contdisc']):
        raise ValueError('Source model/learning config differs from signed identity')
    verify_dependency_versions(identity['packages'])
    code = verify_producer_code(identity['code_sha256'], warm_start=warm_start)
    return dict(path=str(path), manifest_sha256=file_hash(path/'manifest.json'),
                identity=identity, identity_hash=tag['identity_hash'], config=config,
                step=meta['step'], diagnostic=diagnostic, producer_code=code)


class LiveOracleTraining:
    """Shared authenticated TRAIN binding without an offline Replay catalog.

    Constructed from the common checked loader, not a replacement Oracle or
    an invented historical dataset identity. Supported only for diagnostic
    parameter warm-start; exact resume stays with the original FullCheckpoint.
    """
    def __init__(self, query, env):
        from gym_trading_env.research.fg_s3_oracle_state_query import OracleStateQuery, rules_identity
        from gym_trading_env.research.fg_replay import check_files
        if (not isinstance(query, OracleStateQuery) or query.identity['fold'] != 'TRAIN'
                or query.identity['clock_protocol'] != 'S3_MARKET_MINUTE'
                or query.rules != rules_identity(env)):
            raise ValueError('Live source requires the authenticated market-minute TRAIN query')
        source = query.identity['source']
        check_files(source['receipt_files'])
        check_files(source['candidate_files'])
        self.binding = signed(dict(protocol='FG_S3_LIVE_ORACLE_TRAIN_BINDING_V1',
            query_identity=query.identity, query_identity_hash=query.identity_hash,
            clock_protocol='S3_MARKET_MINUTE', diagnostic=True,
            source=dict(training_hash=source['training_hash'], dataset_sha256=None,
                        reward_params_hash=env.reward_config.hash),
            observation_identity=env.observation_identity))
        self.manifest = dict(self.binding)
        self.manifest_sha256 = None  # No historical catalog is being claimed.


def inspect_source(kind, *, catalog=None, live_oracle=None, path=None, seed=None, config=None,
                   base_checkpoint=None, features=None, diagnostic=False, gate_protocol=None,
                   agent_extension=None, composition=None, e_binding=None):
    if composition is not None:
        composition = require_ready(composition)
    return _inspect_source(kind, catalog=catalog, live_oracle=live_oracle, path=path,
        seed=seed, config=config, base_checkpoint=base_checkpoint, features=features,
        diagnostic=diagnostic, gate_protocol=gate_protocol, agent_extension=agent_extension,
        composition=composition, e_binding=e_binding)


def _inspect_source(kind, *, catalog=None, live_oracle=None, path=None, seed=None, config=None,
                    base_checkpoint=None, features=None, diagnostic=False, gate_protocol=None,
                    agent_extension=None, composition=None, e_binding=None):
    """Shared admission core. Waiting E is tested only with bound CPU fixtures."""
    if composition is not None:
        composition = validate_composition(composition)
        if composition['use_b2'] != (agent_extension is not None):
            raise ValueError('Composition and B2 admission extension differ')
    is_e = bool(composition is not None and composition['use_e'])
    if is_e:
        if kind != 'direct_init' or config is None or path is not None:
            raise ValueError('E architecture requires direct_init; old Encoder warm-start forbidden')
        if type(seed) is not int:
            raise ValueError('E direct_init requires an explicit integer seed')
        from embodied.run.fg_s3_e_source import validate_binding
        validate_binding(e_binding, plain(config), composition, diagnostic)
    elif e_binding is not None or (config is not None and config['agent']['enc']['typ'] != 'simple'):
        raise ValueError('E fresh-init requires matching Composition identity')
    if (catalog is None) == (live_oracle is None):
        raise ValueError('Source requires exactly one catalog or authenticated live TRAIN binding')
    if live_oracle is not None:
        if not isinstance(live_oracle, LiveOracleTraining) or kind != 'warm_start' or not diagnostic:
            raise ValueError('Live TRAIN binding supports only explicit diagnostic parameter warm-start')
        from gym_trading_env.research.fg_replay import check_files
        check_files(live_oracle.binding['query_identity']['source']['receipt_files'])
        check_files(live_oracle.binding['query_identity']['source']['candidate_files'])
        catalog = live_oracle
    if agent_extension is not None:
        from dreamerv3.fg_s3_b2_outcome import b2_extensions
        if (kind != 'warm_start' and not (is_e and kind == 'direct_init')) or agent_extension != b2_extensions(agent_extension):
            raise ValueError('Architecture extension requires the frozen B2 parameter warm-start contract')
    if type(diagnostic) is not bool:
        raise ValueError('diagnostic must be an explicit boolean')
    if kind == 'resume_s3c':
        raise ValueError('Resume must use the same run FullCheckpoint, not source transfer')
    if kind not in ('direct_init', 'direct_full', 'after_bc_s3a', 'warm_start'):
        raise ValueError('NOT_SUPPORTED: '+str(kind))
    if kind == 'direct_init':
        if path is not None or seed is None or config is None:
            raise ValueError('direct_init requires explicit seed and config, no checkpoint')
        cfg = plain(config)
        if cfg['seed'] != seed or cfg['agent']['horizon'] != 333 or not cfg['agent']['contdisc']:
            raise ValueError('Initialization seed/Reward horizon differs')
        origin = dict(config=cfg, step=0,
                      diagnostic=bool(catalog.manifest['diagnostic'] or diagnostic))
    elif kind in ('direct_full', 'warm_start'):
        if path is None:
            raise ValueError('direct_full requires a source checkpoint')
        origin = _full(path, catalog, warm_start=kind == 'warm_start')
    else:
        if path is None:
            raise ValueError('after_bc_s3a requires an atomic S3-A package')
        path = Path(path).resolve()
        audited = audit_bc(path)
        state = pickle.loads((path/'state.pkl').read_bytes())
        identity = audited['identity']
        if identity['label_manifest_sha256'] != catalog.manifest['labels_manifest_sha256']:
            raise ValueError('BC/catalog label identity mismatch')
        inherited = _managed_bc_admission(path, identity, catalog)
        reference_identity = (inherited['origin']['identity_hash'] if inherited is not None
                              else identity['source_identity_hash'])
        fm = None
        if features:
            fm = read_manifest(Path(features)/'manifest.json')
            if (file_hash(Path(features)/'manifest.json') != identity['feature_manifest_sha256']
                    or fm['source']['manifest_sha256'] != identity['source_manifest_sha256']
                    or fm['source']['identity_hash'] not in {
                        reference_identity, identity['source_identity_hash']}):
                raise ValueError('BC feature/source provenance mismatch')
            if base_checkpoint is None and Path(fm['source']['path']).is_dir():
                base_checkpoint = fm['source']['path']
        if base_checkpoint:
            base = _full(base_checkpoint, catalog)
            if base['manifest_sha256'] != identity['source_manifest_sha256'] or base['identity_hash'] != reference_identity:
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
                      checkpoint_step=state['bc_updates'], checkpoint_step_unit='bc_gradient_updates',
                      environment_steps=base.get('environment_steps') if base else producer.get('environment_steps'),
                      diagnostic=identity['diagnostic'], baseline=base,
                      baseline_optimizer_equality='PENDING' if base else 'NOT_VERIFIED')
        if inherited is not None:
            if (base is None or inherited['origin']['manifest_sha256'] != base['manifest_sha256']
                    or inherited['origin']['identity_hash'] != base['identity_hash']):
                raise ValueError('Managed BC admission differs from audited baseline')
            from dreamerv3.fg_s3a_actor import parameter_digests
            if parameter_digests(prepare_transfer(inherited)['params']) != state['source_digests']:
                raise ValueError('BC source parameters differ from admitted Gate transfer')
            origin.update(config=inherited['target_config'], inherited_admission=inherited)
    if diagnostic or catalog.manifest['diagnostic']:
        origin['diagnostic'] = True
    from embodied.run.fg_s3_gate import gate_policy, target_config, transfer_policy
    base = origin.get('baseline') if kind == 'after_bc_s3a' else origin
    source_opening = base.get('identity', {}).get('mask_gate_opening') if base else None
    inherited = origin.get('inherited_admission')
    if inherited is not None:
        source_opening = inherited['gate_policy']['opening']
    source_policy = gate_policy(origin['config'], diagnostic=origin['diagnostic'],
                                source_opening=source_opening, protocol=gate_protocol)
    source_config = plain(origin['config'])
    if kind == 'warm_start':
        source_config['agent']['s3_clock_protocol'] = 'S3_MARKET_MINUTE'
    cfg = target_config(source_config, plain(config) if config is not None else None)
    if cfg['agent'].get('s3_clock_protocol') != 'S3_MARKET_MINUTE':
        raise ValueError('New S3 admission requires explicit S3_MARKET_MINUTE model semantics')
    if composition is not None:
        validate_model_composition(composition, cfg['agent']['enc']['typ'])
    policy = gate_policy(cfg, diagnostic=origin['diagnostic'], source_opening=source_opening,
                         protocol=gate_protocol)
    fields = dict(protocol=PROTOCOL, kind=kind, origin=origin,
        target_config=cfg, source_gate_policy=source_policy, gate_policy=policy,
        catalog_manifest_sha256=catalog.manifest_sha256,
        dataset_sha256=catalog.manifest['source']['dataset_sha256'],
        reward_params_hash=catalog.manifest['source']['reward_params_hash'],
        observation_identity=catalog.manifest['observation_identity'],
        optimizer_transfer_policy=('reset_for_market_minute' if kind == 'warm_start' else
            'retain_base_joint_opt' if kind == 'after_bc_s3a' else 'original_direct'),
        clock_protocol='S3_MARKET_MINUTE', target_update_offset=0 if kind == 'warm_start' else origin['step'],
        code_sha256=file_hash(__file__), packages=runtime_versions())
    if live_oracle is not None:
        fields.update(teacher_source_kind='live_oracle', live_training_binding=live_oracle.binding)
    if agent_extension is not None:
        fields['agent_extension'] = agent_extension
    if composition is not None:
        fields[KEY] = composition
    if is_e:
        fields['e_binding'] = e_binding
    initial = signed(fields)
    state = _prepare_transfer(initial)
    if state is not None:
        from dreamerv3.fg_s3a_actor import parameter_digests
        fields['source_param_digests'] = parameter_digests(state['params'])
    enabled = bool(origin['config']['agent']['avail_actor_enabled']) if state is None else bool(
        np.asarray(state['params']['avail_actor_gate/value']))
    control = _source_control_state(initial)
    if not control['complete'] and not origin['diagnostic']:
        raise ValueError('Formal source lacks Gate controller/window state')
    _, gate_transfer = transfer_policy(source_policy, policy, control, enabled)
    if kind == 'warm_start':
        gate_transfer.update(reset_required=True, reset_reason='new_market_minute_run',
            new_gate_state=dict(actor_enabled=False, gate=None, samples=0), new_window_present=False)
    model_identity = dict(protocol='FG_S3_ADMITTED_MODEL_V1', kind=kind,
        source_manifest_sha256=origin.get('manifest_sha256'),
        source_identity_hash=origin.get('identity_hash'),
        source_param_digests=fields.get('source_param_digests'),seed=cfg['seed'],
        agent_config=cfg['agent'],gate_policy=policy,gate_transfer=gate_transfer,
        admission_code_sha256=file_hash(__file__),
        gate_code_sha256=file_hash(Path(__file__).with_name('fg_s3_gate.py')))
    if agent_extension is not None:
        model_identity['agent_extension'] = agent_extension
    if composition is not None:
        model_identity[KEY] = composition
    if is_e:
        model_identity['e_binding'] = e_binding
        model_identity['target_training'] = dict(
            reward_params_hash=fields['reward_params_hash'],
            teacher_observation_identity=fields['observation_identity'],
            observation_contract_hash=e_binding['observation']['contract_hash'],
            data_manifest_sha256=e_binding['data_manifest_sha256'],
            training_hash=catalog.manifest['source']['training_hash'], clock_protocol=fields['clock_protocol'])
    if kind == 'warm_start':
        model_identity['target_training'] = dict(
            reward_params_hash=fields['reward_params_hash'], observation_identity=fields['observation_identity'],
            training_hash=catalog.manifest['source']['training_hash'], clock_protocol=fields['clock_protocol'])
        if live_oracle is not None:
            model_identity['target_training']['live_binding_hash'] = live_oracle.binding['content_hash']
    return signed(dict(fields, gate_transfer=gate_transfer,target_model_identity=model_identity,
                       target_model_identity_hash=fingerprint(model_identity)))


def _managed_bc_admission(path, identity, catalog):
    """Read the hash-bound managed snapshot; never infer BC threshold from a name."""
    from gym_trading_env.research.fg_replay import check_files
    root = path.parent.parent
    snapshot = root/'managed_snapshot'
    if not snapshot.exists():
        return None
    manifest = read_manifest(snapshot/'manifest.json')
    if manifest['protocol'] != 'FG_S3_RUN_SNAPSHOT_V1':
        raise ValueError('Foreign managed BC snapshot')
    check_files({str(snapshot/n):h for n,h in manifest['files'].items()})
    preparation = read_manifest(snapshot/'preparation_receipt.json')
    if (manifest['binding']['prepared_manifest_sha256'] != file_hash(snapshot/'preparation_receipt.json')
            or identity['run_id'] != preparation['experiment']+'_'+preparation['content_hash'][:16]
            or read_manifest(root/'run_identity.json')['identity'] != identity):
        raise ValueError('BC training identity is not bound to its preparation')
    name = 'source_admission_receipt.json'
    if name not in manifest['files']:
        return None  # Historical managed BC before shared source admission.
    if (preparation['files'][name] != file_hash(snapshot/name)
            or manifest['binding']['feature_binding_sha256'] != file_hash(snapshot/'features_binding.json')):
        raise ValueError('Managed BC source/feature binding differs')
    binding = read_manifest(snapshot/'features_binding.json')
    spec = read_manifest(snapshot/name)
    if (binding['feature_manifest_sha256'] != identity['feature_manifest_sha256']
            or spec['catalog_manifest_sha256'] != catalog.manifest_sha256
            or spec['kind'] != 'direct_full'
            or spec['origin']['manifest_sha256'] != identity['source_manifest_sha256']):
        raise ValueError('Managed BC shared source/Catalog differs')
    if spec['protocol'] == 'FG_S3_SOURCE_ADMISSION_V1':
        return None  # Historical receipt did not change the baseline threshold.
    if spec['protocol'] != PROTOCOL:
        raise ValueError('Foreign managed BC admission protocol')
    if (spec['target_model_identity_hash'] != identity['source_identity_hash']
            or fingerprint(spec['target_model_identity']) != identity['source_identity_hash']):
        raise ValueError('BC admitted target model identity differs')
    return spec


def prepare_transfer(spec, target_schema=None):
    if KEY in spec:
        composition = require_ready(spec[KEY])
        if 'target_model_identity' in spec and require_ready(spec['target_model_identity'].get(KEY)) != composition:
            raise ValueError('S3_COMPOSITION_IDENTITY_MISMATCH')
    return _prepare_transfer(spec, target_schema)


def _prepare_transfer(spec, target_schema=None):
    if KEY in spec:
        composition = validate_model_composition(spec[KEY], spec['target_config']['agent']['enc']['typ'])
        if 'target_model_identity' in spec and spec['target_model_identity'].get(KEY) != composition:
            raise ValueError('S3_COMPOSITION_IDENTITY_MISMATCH')
        if composition['use_e']:
            from embodied.run.fg_s3_e_source import validate_binding, validate_e_records
            validate_binding(spec.get('e_binding'), spec['target_config'], composition, spec['origin']['diagnostic'])
            if spec['kind'] != 'direct_init':
                raise ValueError('E architecture requires direct_init')
            if 'target_model_identity' in spec:
                identity = spec['target_model_identity']
                validate_e_records(spec, identity)
                if (spec.get('target_model_identity_hash') != fingerprint(identity)
                        or identity['agent_config'] != spec['target_config']['agent']):
                    raise ValueError('E admitted model identity/config hash mismatch')
    if spec['protocol'] != PROTOCOL or fingerprint({k:v for k,v in spec.items() if k!='content_hash'}) != spec['content_hash']:
        raise ValueError('Source admission receipt corrupt')
    origin = spec['origin']
    if spec['kind'] == 'direct_init':
        return None
    path = Path(origin['path'])
    if file_hash(path/'manifest.json') != origin['manifest_sha256']:
        raise ValueError('Source changed after admission')
    if spec['kind'] in ('direct_full', 'warm_start'):
        validate_bundle(path)
        state = pickle.loads((path/'agent.pkl').read_bytes())
    else:
        audit_bc(path)
        state = pickle.loads((path/'state.pkl').read_bytes())['agent']
    required = {'params','counters','policy_params','pending_sync','pending_outs','pending_mets'}
    if set(state) != required or state['counters']['updates'] != origin['step']:
        raise ValueError('Incomplete full Agent state/counters')
    if origin.get('agent_counters') is not None and state['counters'] != origin['agent_counters']:
        raise ValueError('Online frozen counter evidence differs from actual Agent')
    if any('s3a_bc_opt' in k or 's3b_bc_opt' in k for k in state['params']):
        raise ValueError('Independent BC optimizer in Agent state')
    for k, v in state['params'].items():
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite source parameter: '+k)
    if 'source_param_digests' in spec:
        from dreamerv3.fg_s3a_actor import parameter_digests
        if parameter_digests(state['params']) != spec['source_param_digests']:
            raise ValueError('Source parameter evidence changed after admission')
    if target_schema is not None:
        actual = {k:(v.shape,str(v.dtype)) for k,v in state['params'].items()}
        extension = spec.get('agent_extension')
        if spec['kind'] == 'warm_start' and extension is not None:
            prefixes = ('b2_event/', 'b2_early/', 'b2_adapter/', 'b2_schedule/')
            extras = set(target_schema) - set(actual)
            import ninjax as nj
            # nj.Tree stores optax leaves using nj.flatten's canonical names,
            # which remove underscores (b2_event -> b2event). Derive exact
            # permitted suffixes from real target NN keys, not a loose regex.
            encoded = set(nj.flatten({k:0 for k in target_schema if k.startswith(prefixes[:3])})[0])
            allowed = lambda k: k.startswith(prefixes) or (
                k.startswith('opt/state/') and any(k.endswith('/'+name) for name in encoded))
            if (set(actual) - set(target_schema) or any(actual[k] != target_schema[k] for k in actual)
                    or any(not allowed(k) for k in extras)
                    or not all(any(k.startswith(p) for k in extras) for p in prefixes)):
                detail = dict(missing=sorted(set(actual)-set(target_schema)),
                    changed={k:[actual[k],target_schema[k]] for k in actual if k in target_schema and actual[k]!=target_schema[k]},
                    unexpected=sorted(k for k in extras if not allowed(k)),
                    missing_extensions=[p for p in prefixes if not any(k.startswith(p) for k in extras)])
                raise ValueError('B2 extension changed original parameter schema or added foreign state: '+json.dumps(detail))
        elif actual != target_schema:
            raise ValueError('Source parameter keys/dtypes/shapes differ from target')
    if spec.get('gate_transfer', {}).get('reset_required'):
        state = dict(state, params=dict(state['params']))
        key = 'avail_actor_gate/value'
        state['params'][key] = np.zeros_like(state['params'][key])
        if state['policy_params'] is not None:
            state['policy_params'] = dict(state['policy_params'])
            if key in state['policy_params']:
                state['policy_params'][key] = np.zeros_like(state['policy_params'][key])
    return state


def freeze_source(path, output):
    """Copy one published node and its identity, hash before/after, never hardlink."""
    import shutil
    import os
    from gym_trading_env.research.fg_s3_teacher_catalog import guard_output
    from gym_trading_env.research.fg_replay import write_json
    source = Path(path).resolve()
    meta = validate_bundle(source)
    initial = file_hash(source/'manifest.json')
    root = guard_output(output, (source.parent.parent,))
    root.mkdir(parents=True)
    dest = root/'ckpt'/source.name
    shutil.copytree(source, dest)
    online = meta['identity'].get('schema') == 1 and 'code_hash' in meta['identity']
    extras = {}
    if online:
        from dreamerv3.configuration import read_config_yaml
        config = read_config_yaml(dest/'config.yaml')
        mapping = _online_code(config, meta['identity']['code_hash'])
        (root/'producer_code_hashes.json').write_text(json.dumps(mapping,sort_keys=True))
        for name,src in (('entry_manifest.json',config['fg_entry']['manifest']),
                         ('online_transfer.json',config['run']['fg_stage_transfer']),
                         ('selector.yaml',config['env']['gymnasium']['config_path'])):
            shutil.copyfile(src,root/name)
        counts = pickle.loads((dest/'agent.pkl').read_bytes())['counters']
        write_json(root/'agent_counters.json',signed(dict(counters=counts)))
        extras = {p.name:file_hash(p) for p in root.iterdir() if p.is_file()}
    else:
        tags = [name for name in ('single3_identity.json','s3a_diagnostic_identity.json')
                if (source.parent.parent/name).is_file()]
        if len(tags) != 1:
            raise ValueError('Source snapshot requires one parent identity')
        shutil.copyfile(source.parent.parent/tags[0],root/tags[0])
    validate_bundle(dest)
    validate_bundle(source)
    if file_hash(source/'manifest.json') != initial or file_hash(dest/'manifest.json') != initial:
        raise ValueError('Source changed during snapshot copy')
    evidence = dict(format='FG_ONLINE_FULL_SOURCE_V1',files=extras) if online else dict(
        parent_identity_sha256=file_hash(root/tags[0]))
    receipt = signed(dict(protocol=PROTOCOL, source=str(source), snapshot=str(dest),
        manifest_sha256=initial, **evidence))
    write_json(root/'source_snapshot.json',receipt)
    for p in root.rglob('*'):
        if p.is_file(): os.chmod(p,0o444)
    return receipt


def _source_control_state(spec):
    origin = spec['origin']
    if origin.get('inherited_admission') is not None:
        return transfer_control_state(origin['inherited_admission'])
    base = origin.get('baseline') if spec['kind']=='after_bc_s3a' else origin
    if spec['kind']=='direct_init':
        return dict(complete=True, new_experiment=True, gate=None, aggregate=None, samples=0,
                    opening=spec['gate_policy']['opening'])
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


def transfer_control_state(spec):
    """Apply the admitted new-run policy; exact resume never calls this boundary."""
    from embodied.run.fg_s3_gate import transfer_policy
    state = _source_control_state(spec)
    if spec['kind'] == 'warm_start':
        return dict(complete=True, new_experiment=True, gate=None, aggregate={}, samples=0,
                    opening=spec['gate_policy']['opening'])
    enabled = spec['gate_transfer']['original_gate_state']['actor_enabled']
    control, proof = transfer_policy(spec['source_gate_policy'], spec['gate_policy'], state, enabled)
    if proof != spec['gate_transfer']:
        raise ValueError('Source Gate control changed after admission')
    return control
