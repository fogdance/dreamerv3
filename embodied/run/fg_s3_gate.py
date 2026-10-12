"""Shared Gate policy; new-run transfers and same-run restoration are distinct."""
import copy
import elements

from embodied.run.action_mask_warmup import ActionMaskWarmup
from embodied.run.fg_checkpoint import plain

SINGLE3 = 'FG_SINGLE3_MASK_GATE_V1'
DIAGNOSTIC = 'FG_S3_DIAGNOSTIC_CONFIGURED_GATE_V1'


def gate_policy(config, *, diagnostic, source_opening=None, protocol=None):
    from dreamerv3.agent import validate_availability_threshold
    threshold = validate_availability_threshold(config['agent']['avail_threshold'])
    protocol = protocol or (DIAGNOSTIC if diagnostic else SINGLE3)
    if protocol not in (SINGLE3, DIAGNOSTIC) or (not diagnostic and protocol != SINGLE3):
        raise ValueError('Formal Single3 requires the Single3 Gate protocol')
    cfg = elements.Config(config['action_mask_warmup'])
    factory = ActionMaskWarmup.for_single3_1m if protocol == SINGLE3 else ActionMaskWarmup
    controller = factory(cfg, False)
    if protocol == SINGLE3:
        if (not controller.add_recall_enabled
                or controller.min_add_positives < 16
                or controller.min_train_samples < 25000
                or controller.regression_limit != 5 or not controller.stop_on_regression
                or controller.thresholds['post_add_recall'] < .8
                or controller.thresholds['prior_add_recall'] < .8):
            raise ValueError('Single3 ADD support, recall or regression Gate policy changed')
        for name, minimum in dict(post_accuracy=.995,post_exact=.990,
                                  prior_accuracy=.990,prior_exact=.980).items():
            if controller.thresholds[name] < minimum:
                raise ValueError('Single3 accuracy Gate policy weakened: '+name)
        if controller.thresholds['prior_fpr'] > .005 or controller.thresholds['img_fallback'] > .0001:
            raise ValueError('Single3 FPR/fallback Gate policy weakened')
        if source_opening is not None and (
                source_opening['thresholds'] != controller.opening_thresholds
                or source_opening['consecutive_reports'] != controller.required_passes):
            raise ValueError('Source opening differs from the Single3 twenty-report Gate policy')
    elif source_opening:
        controller.opening_thresholds = dict(source_opening['thresholds'])
        controller.required_passes = source_opening['consecutive_reports']
    return dict(protocol=protocol, classification_threshold=threshold,
                classification_rule='float32_ge_half_legacy_boundary_v1',
                opening=dict(thresholds=controller.opening_thresholds,
                             consecutive_reports=controller.required_passes),
                source_opening=source_opening, config=plain(cfg))


def target_config(source, requested=None):
    """Only availability threshold may change inside the inherited Agent config."""
    cfg = copy.deepcopy(source if requested is None else requested)
    agent = dict(cfg['agent']);agent['avail_threshold'] = source['agent']['avail_threshold']
    if (agent != source['agent']
            or cfg['action_mask_warmup'] != source['action_mask_warmup']
            or cfg['jax']['compute_dtype'] != source['jax']['compute_dtype']):
        raise ValueError('Target changed source network/optimizer/loss/Gate policy/compute dtype')
    from dreamerv3.agent import validate_availability_threshold
    validate_availability_threshold(cfg['agent']['avail_threshold'])
    return cfg


def transfer_policy(source_policy, target_policy, source_control, source_enabled):
    """Authenticate old statistics, then fail closed on a changed classification."""
    control = copy.deepcopy(source_control)
    if control['gate'] is not None:
        state = control['gate']
        if (bool(state['actor_enabled']) != bool(source_enabled)
                or any(type(state[k]) is not int or state[k] < 0 for k in
                       ('reports','consecutive_passes','consecutive_regressions'))
                or state['consecutive_passes'] > state['reports']
                or state['consecutive_regressions'] > state['reports']):
            raise ValueError('Source Gate model/controller or statistics disagree')
        if (state['actor_enabled'] and state['switched_step'] < 0
                or not state['actor_enabled'] and state['switched_step'] != -1):
            raise ValueError('Source Gate activation chronology invalid')
    if type(control['samples']) is not int or control['samples'] < 0:
        raise ValueError('Source Gate sample window invalid')
    changed = (source_policy['classification_threshold'] != target_policy['classification_threshold']
               or source_policy['protocol'] != target_policy['protocol'])
    old = dict(actor_enabled=bool(source_enabled), gate=control['gate'], samples=control['samples'])
    if changed:
        factory = ActionMaskWarmup.for_single3_1m if target_policy['protocol'] == SINGLE3 else ActionMaskWarmup
        controller = factory(elements.Config(target_policy['config']), False)
        control.update(gate=controller.save(), aggregate={}, samples=0,
                       opening=target_policy['opening'])
    return control, dict(source_gate_threshold=source_policy['classification_threshold'],
        target_gate_threshold=target_policy['classification_threshold'],
        source_gate_protocol=source_policy['protocol'], target_gate_protocol=target_policy['protocol'],
        reset_required=changed, reset_reason='classification_threshold_or_protocol_changed' if changed else None,
        original_gate_state=old, new_gate_state=dict(actor_enabled=False if changed else bool(source_enabled),
            gate=control['gate'], samples=control['samples']),
        original_window_present=bool(source_control['aggregate']),
        new_window_present=bool(control['aggregate']))


def make_gate_controller(config, policy, control, actor_enabled):
    """Build from an admitted transfer state, or a fresh state before full resume."""
    expected = gate_policy(config, diagnostic=policy['protocol'] == DIAGNOSTIC,
                           source_opening=policy['source_opening'], protocol=policy['protocol'])
    if expected != policy:
        raise ValueError('Gate policy receipt/config differs')
    factory = ActionMaskWarmup.for_single3_1m if policy['protocol'] == SINGLE3 else ActionMaskWarmup
    controller = factory(elements.Config(config['action_mask_warmup']), False)
    controller.opening_thresholds = dict(policy['opening']['thresholds'])
    controller.required_passes = policy['opening']['consecutive_reports']
    if control['gate'] is not None:
        controller.load(control['gate'])
    else:
        controller.actor_enabled = actor_enabled
    if controller.actor_enabled != actor_enabled:
        raise ValueError('Source model/controller Gate disagree')
    return controller
