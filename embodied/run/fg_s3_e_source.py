"""E fresh-init identity for synthetic and reviewed real TRAIN data.

Public E model activation remains gated until the entire M4 lifecycle passes.
"""
from pathlib import Path
import subprocess
import json

from embodied.run.fg_checkpoint import file_hash
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, signed
from gym_trading_env.research.fg_s3_e_observation import observation_contract
from dreamerv3.fg_s3_composition import validate_model_composition

PROTOCOL = 'FG_S3_E_SOURCE_BINDING_V1'
SYNTHETIC = 'FG_S3_E_MODEL_SYNTHETIC_V1'
REAL = 'S3_E_DATA_V1'
REAL_SCOPE = 'CPU_REAL_TRAIN_M4_NOT_FORMAL_RELEASED'


def source_code():
    from embodied.run.fg_s3_v2_train import execution_code
    return execution_code()


def make_binding(config, composition, data_manifest):
    """Bind reviewed E dataset identity; public training remains waiting_e.

    Real E manifests are content-addressed collection indexes, not the legacy
    `signed(content_hash)` synthetic fixture format. They require verification
    of every immutable TRAIN bundle before the CPU M4 contract can use them.
    """
    composition = validate_model_composition(composition, config['agent']['enc']['typ'])
    path = Path(data_manifest).resolve(strict=True)
    data = json.loads(path.read_text())
    if data.get('schema') == REAL and data.get('kind') == 'collection':
        if path.name != 'index_manifest.json':
            raise ValueError('E real data requires its canonical collection index')
        from e_data_v1.io import verify_collection
        review = verify_collection(path.parent)
        if review['status'] != 'PASS':
            raise ValueError('E real TRAIN collection verification failed')
        scope, data_id = REAL_SCOPE, data['index_sha256']
        schema_hash, pool_hash = data['schema_sha256'], data['source_pool_sha256']
    elif data.get('schema') == SYNTHETIC:
        read_manifest(path)  # full legacy signed synthetic verification
        scope, data_id = 'CPU_SYNTHETIC_CONTRACT_ONLY', data['content_hash']
        schema_hash, pool_hash = None, None
    else:
        raise ValueError('Unknown or unsigned E data source manifest')
    dreamer = Path(__file__).resolve().parents[2]
    gym = dreamer.parent/'gym-trading-env'
    data_code = {str(p.relative_to(gym)): file_hash(p) for p in (
        gym/'e_data_v1/core.py', gym/'e_data_v1/provider.py',
        gym/'e_data_v1/schema.py', gym/'e_data_v1/reference.py') if p.is_file()}
    if len(data_code) != 4:
        raise ValueError('E data runtime implementation is incomplete')
    versions = {str(r): subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
                for r in (dreamer.parent/'gym-trading-env', dreamer)}
    dirty = {str(r): subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).splitlines()
             for r in (dreamer.parent/'gym-trading-env', dreamer)}
    return signed(dict(protocol=PROTOCOL, scope=scope,
        observation=observation_contract(), data_manifest=str(path),
        data_manifest_sha256=file_hash(path), data_schema=data['schema'],
        data_identity=data_id, data_schema_sha256=schema_hash,
        data_source_pool_sha256=pool_hash, data_runtime_code=data_code, seed=config['seed'],
        agent_config_hash=fingerprint(config['agent']),
        encoder=composition['encoder'], teaching=composition['teaching'],
        versions=versions, dirty=dirty, code=source_code()))


def validate_e_records(*documents):
    bindings = [d.get('e_binding') for d in documents]
    if any(x is None for x in bindings) or any(x != bindings[0] for x in bindings):
        raise ValueError('E source/prepared/model/run data identity mismatch')
    return bindings[0]


def validate_binding(value, config, composition, diagnostic):
    if not diagnostic or not isinstance(value, dict):
        raise ValueError('E fresh-init requires a diagnostic authenticated data binding')
    if value != make_binding(config, composition, value.get('data_manifest', '')):
        raise ValueError('E fresh-init seed/config/schema/code/data identity mismatch')
    agent = config['agent']
    if (type(config['seed']) is not int or config['seed'] < 0
            or agent.get('s3_clock_protocol') != 'S3_MARKET_MINUTE'
            or not agent['contdisc'] or agent['horizon'] != 333
            or agent['avail_actor_enabled'] or config['replay_context'] != 1
            or agent['replay_context_mode'] != 'recompute_prefix'):
        raise ValueError('E fresh-init requires fresh CLOSED Gate and market-minute H333')
    return value
