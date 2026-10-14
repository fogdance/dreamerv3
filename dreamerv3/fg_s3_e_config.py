"""Explicit fresh-init configuration using the canonical Dreamer presets."""
import copy
from pathlib import Path

import elements
from .configuration import read_config_yaml

ENCODER = 'e_multiscale_1m15m'


def e_target_config(config):
    cfg = copy.deepcopy(config)
    minute = cfg['agent']['enc']['simple']
    cfg['agent']['enc']['typ'] = ENCODER
    cfg['agent']['enc'][ENCODER] = {k: minute[k] for k in
        ('units', 'norm', 'act', 'symlog', 'winit')}
    cfg['agent']['enc'][ENCODER].update(layers=3, context_units=128)
    cfg['agent'].update(s3_clock_protocol='S3_MARKET_MINUTE',
        replay_context_mode='recompute_prefix', horizon=333, contdisc=True)
    cfg['replay_context'] = 1
    return cfg


def resolve_model_init(value):
    if not isinstance(value, dict) or set(value) != {'seed', 'presets'}:
        raise ValueError('model_init requires explicit seed and presets')
    seed, presets = value['seed'], value['presets']
    if type(seed) is not int or seed < 0 or not isinstance(presets, list) or not presets:
        raise ValueError('Invalid fresh-init seed/presets')
    definitions = read_config_yaml(Path(__file__).with_name('configs.yaml'))
    cfg = elements.Config(definitions['defaults'])
    for preset in presets:
        if not isinstance(preset, str) or preset not in (
                'size1m', 'size12m', 'size25m', 'size50m', 'size100m', 'size200m', 'size400m'):
            raise ValueError('Fresh-init requires an explicit Dreamer size preset')
        cfg = cfg.update(definitions[preset])
    cfg = cfg.update({'seed': seed, 'dreamer.seed': seed,
        'agent.avail_threshold': .30, 'agent.avail_actor_enabled': False,
        'agent.avail_always_available_index': 0,
        'action_mask_warmup.enabled': True, 'action_mask_warmup.add_recall_enabled': True})
    return e_target_config(dict(cfg))
