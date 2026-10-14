"""E fresh-init identity. Synthetic bindings are CPU contract evidence only."""
from pathlib import Path
import subprocess

from embodied.run.fg_checkpoint import file_hash
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, signed
from gym_trading_env.research.fg_s3_e_observation import observation_contract
from dreamerv3.fg_s3_composition import validate_model_composition

PROTOCOL = 'FG_S3_E_SOURCE_BINDING_V1'
SYNTHETIC = 'FG_S3_E_MODEL_SYNTHETIC_V1'


def source_code():
    from embodied.run.fg_s3_v2_train import execution_code
    return execution_code()


def make_binding(config, composition, data_manifest):
    """Bind frozen synthetic fixtures; real E-Data admission needs M4 Review."""
    composition = validate_model_composition(composition, config['agent']['enc']['typ'])
    path = Path(data_manifest).resolve()
    data = read_manifest(path)
    if data.get('schema') != SYNTHETIC:
        raise ValueError('E_DATA_REVIEW_REQUIRED: real data admission is not unlocked')
    dreamer = Path(__file__).resolve().parents[2]
    versions = {str(r): subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
                for r in (dreamer.parent/'gym-trading-env', dreamer)}
    dirty = {str(r): subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).splitlines()
             for r in (dreamer.parent/'gym-trading-env', dreamer)}
    return signed(dict(protocol=PROTOCOL, scope='CPU_SYNTHETIC_CONTRACT_ONLY',
        observation=observation_contract(), data_manifest=str(path),
        data_manifest_sha256=file_hash(path), data_schema=data['schema'],
        data_identity=data['content_hash'], seed=config['seed'],
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
