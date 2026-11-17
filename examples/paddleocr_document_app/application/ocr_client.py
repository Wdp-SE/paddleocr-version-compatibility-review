"""Author-owned runtime; submitted examples are never executed by the service."""
import hashlib
import json
import math
from pathlib import Path


def _score(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError('drop_score must be a finite number between 0 and 1')
    return float(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate configuration field: ' + key)
        result[key] = value
    return result


def load_configuration(config_path=None, *, drop_score=None):
    """Load at most 16 KiB of explicit JSON. No imports, env overrides or YAML."""
    declared = {}
    config_hash = None
    source = None
    if config_path is not None:
        path = Path(config_path).resolve(strict=True)
        with path.open('rb') as handle:
            content = handle.read(16385)
        if len(content) > 16384:
            raise ValueError('configuration exceeds 16 KiB')
        try:
            declared = json.loads(content.decode('utf-8'), object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError('configuration must be UTF-8 JSON') from exc
        if not isinstance(declared, dict) or set(declared) - {'schema_version', 'drop_score'}:
            raise ValueError('configuration must contain only schema_version and drop_score')
        if 'schema_version' in declared and (type(declared['schema_version']) is not int or declared['schema_version'] != 1):
            raise ValueError('unsupported configuration schema')
        if 'drop_score' in declared:
            _score(declared['drop_score'])
        config_hash = hashlib.sha256(content).hexdigest()
        source = str(path)
    effective = {'lang': 'ch', 'drop_score': _score(declared.get('drop_score', 0.5))}
    sources = {'lang': 'default', 'drop_score': 'json' if 'drop_score' in declared else 'default'}
    if drop_score is not None:
        effective['drop_score'] = _score(drop_score)
        sources['drop_score'] = 'explicit_override'
    return {'config_sha256': config_hash, 'config_source': source,
            'declared_values': declared, 'effective_values': effective, 'value_sources': sources}


def create_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang='ch')


def create_configured_engine(config_path=None, *, drop_score=None, engine_factory=None, provenance=None):
    configuration = load_configuration(config_path, drop_score=drop_score)
    parameters = {'lang': 'ch'}
    # Preserve the original no-config factory surface for existing static examples.
    if config_path is not None or drop_score is not None:
        parameters['drop_score'] = configuration['effective_values']['drop_score']
    if engine_factory is None:
        from paddleocr import PaddleOCR
        engine_factory = PaddleOCR
    engine = engine_factory(**parameters)
    if provenance is not None:
        provenance.update(configuration, factory_parameters=dict(parameters))
    return engine


def recognize(path):
    engine = create_engine()
    return engine.ocr(path)


def recognize_configured(path, *, config_path=None, drop_score=None, engine_factory=None, provenance=None):
    engine = create_configured_engine(config_path, drop_score=drop_score, engine_factory=engine_factory,
                                      provenance=provenance)
    return engine.ocr(path)
