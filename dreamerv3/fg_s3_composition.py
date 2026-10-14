"""Pure S3 configuration contract. No model imports, I/O or runtime unlocks."""
from collections.abc import Mapping

LEGACY_SCHEMA = 'FG_S3_V2_DIAGNOSTIC_CONFIG_V1'
CONFIG_SCHEMA = 'FG_S3_V2_COMPOSITION_CONFIG_V1'
KEY = 'resolved_composition'
ENCODERS = ('legacy_1m', 'e_multiscale_1m15m')
TEACHING = ('a2', 'tutor_cd', 'b2', 'tutor_cd_b2')
MODES = {'a2': 'a2', 'tutor_cd': 'tutor', 'b2': 'b2', 'tutor_cd_b2': None}


def resolve_composition(config):
    """Parse a raw config/fragment; C2 deliberately stays outside composition."""
    if not isinstance(config, Mapping):
        raise ValueError('S3 composition configuration must be a mapping')
    if KEY in config:
        raise ValueError('resolved_composition is output metadata, not a YAML input')
    if 'composition' in config:
        if 'mode' in config:
            raise ValueError('S3_MODE_COMPOSITION_CONFLICT: select exactly one syntax')
        value = config['composition']
        if not isinstance(value, Mapping) or set(value) != {'encoder', 'teaching'}:
            raise ValueError('composition requires exactly encoder and teaching')
        encoder, teaching = value['encoder'], value['teaching']
    else:
        mode = config.get('mode')
        if mode == 'c2':
            return None
        if not isinstance(mode, str) or mode not in ('a2', 'tutor', 'b2'):
            raise ValueError('Select composition or legacy mode: a2/tutor/b2/c2')
        encoder, teaching = 'legacy_1m', {'a2': 'a2', 'tutor': 'tutor_cd', 'b2': 'b2'}[mode]
    if not isinstance(encoder, str) or encoder not in ENCODERS:
        raise ValueError('Unknown encoder; select '+', '.join(ENCODERS))
    if not isinstance(teaching, str) or teaching not in TEACHING:
        raise ValueError('Unknown teaching; select '+', '.join(TEACHING))
    state = ('waiting_tutor_b2' if teaching == 'tutor_cd_b2' else
             'waiting_e' if encoder != 'legacy_1m' else 'ready')
    return dict(encoder=encoder, teaching=teaching, base_oracle_bc=True,
        use_e=encoder != 'legacy_1m', use_tutor=teaching in ('tutor_cd', 'tutor_cd_b2'),
        use_b2=teaching in ('b2', 'tutor_cd_b2'), legacy_mode=MODES[teaching],
        support_state=state, schema_version=1)


def validate_composition(value):
    """Recompute derived fields; never trust a saved/user-supplied ready flag."""
    if not isinstance(value, Mapping):
        raise ValueError('S3_COMPOSITION_IDENTITY_MISSING')
    expected = resolve_composition({'composition': {
        'encoder': value.get('encoder'), 'teaching': value.get('teaching')}})
    if set(value) != set(expected) or any(type(value[k]) is not type(v) or value[k] != v
                                        for k, v in expected.items()):
        raise ValueError('S3_COMPOSITION_IDENTITY_MISMATCH')
    return expected


def require_ready(value):
    value = validate_composition(value)
    if value['support_state'] == 'waiting_tutor_b2':
        raise ValueError('S3_TUTOR_B2_NOT_IMPLEMENTED')
    if value['support_state'] == 'waiting_e':
        raise ValueError('S3_E_ENCODER_NOT_IMPLEMENTED')
    return value


def validate_encoder(value, encoder_type):
    value = require_ready(value)
    validate_model_composition(value, encoder_type)


def validate_model_composition(value, encoder_type, mode=None):
    """Structural model contract, without declaring a production capability.

    Internal CPU model tests may instantiate a waiting_e architecture. Public
    factories/lifecycle still call require_ready before invoking this core.
    """
    value = validate_composition(value)
    if value['teaching'] == 'tutor_cd_b2':
        raise ValueError('S3_TUTOR_B2_NOT_IMPLEMENTED')
    expected = 'simple' if value['encoder'] == 'legacy_1m' else value['encoder']
    if encoder_type != expected:
        raise ValueError('S3_COMPOSITION_ENCODER_MISMATCH: expected '+expected)
    if mode is not None and mode != value['legacy_mode']:
        raise ValueError('S3_COMPOSITION_MODE_MISMATCH')
    return value


def normalize_config(config, *, executable=True):
    """Versioned raw YAML -> runtime settings; legacy keys stay unchanged."""
    resolved = resolve_composition(config)
    schema = config.get('schema')
    if schema == LEGACY_SCHEMA and 'composition' not in config:
        return dict(config)
    if schema != CONFIG_SCHEMA or 'composition' not in config:
        raise ValueError('Composition requires schema '+CONFIG_SCHEMA)
    if executable:
        require_ready(resolved)
    result = {k: v for k, v in config.items() if k != 'composition'}
    result.update(mode=resolved['legacy_mode'], resolved_composition=resolved)
    return result


def settings_composition(settings):
    """Authenticate normalized settings before any preparation/runtime writes."""
    if settings.get('schema') == CONFIG_SCHEMA:
        if 'composition' in settings:
            raise ValueError('Expected normalized composition settings')
        value = require_ready(settings.get(KEY))
        if settings.get('mode') != value['legacy_mode']:
            raise ValueError('S3_COMPOSITION_MODE_MISMATCH')
        return value
    if KEY in settings or 'composition' in settings:
        raise ValueError('S3_COMPOSITION_SCHEMA_MISMATCH')
    return None


def identity_fields(settings):
    value = settings_composition(settings)
    return {} if value is None else {KEY: value}


def validate_records(settings, *documents):
    """All new identity records must agree; absent legacy fields remain absent."""
    expected = settings_composition(settings)
    for document in documents:
        if expected is None:
            if KEY in document:
                raise ValueError('S3_COMPOSITION_SCHEMA_MISMATCH')
        elif validate_composition(document.get(KEY)) != expected:
            raise ValueError('S3_COMPOSITION_IDENTITY_MISMATCH')
    return expected
