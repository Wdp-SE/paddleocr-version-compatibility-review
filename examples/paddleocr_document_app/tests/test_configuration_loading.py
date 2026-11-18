"""Actual author-owned entry point, replacing only the expensive OCR constructor."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application import ocr_client, pipeline


class ScoreEngine:
    def __init__(self, *, lang, drop_score=0.5):
        assert lang == 'ch'
        self.threshold = drop_score

    def ocr(self, path):
        return [[[None, (text, score)] for text, score in [
            ('below', 0.49), ('boundary', 0.5), ('between', 0.7), ('high', 0.9)
        ] if score >= self.threshold], [], [[None, ('last', 0.95)]]]


def test_no_config_keeps_existing_constructor_call(monkeypatch):
    class VendorModule:
        @staticmethod
        def PaddleOCR(**kwargs):
            assert kwargs == {'lang': 'ch'}
            return ScoreEngine(**kwargs)
    monkeypatch.setitem(sys.modules, 'paddleocr', VendorModule)
    assert pipeline.process('fixture')['pages'][0]['text'] == 'boundary\nbetween\nhigh'


@pytest.mark.parametrize('threshold, expected', [(0.5, 'boundary\nbetween\nhigh'), (0.8, 'high')])
def test_json_is_consumed_by_real_factory_and_changes_downstream(tmp_path, threshold, expected):
    path = tmp_path / 'settings.json'
    content = json.dumps({'schema_version': 1, 'drop_score': threshold}).encode()
    path.write_bytes(content)
    provenance = {}
    result = pipeline.process_configured('fixture', config_path=path, engine_factory=ScoreEngine,
                              provenance=provenance)
    assert result == {'schema_version': 1, 'pages': [
        {'page_index': 0, 'text': expected}, {'page_index': 1, 'text': ''},
        {'page_index': 2, 'text': 'last'}]}
    assert provenance['config_sha256'] == hashlib.sha256(content).hexdigest()
    assert provenance['effective_values'] == {'lang': 'ch', 'drop_score': threshold}
    assert provenance['value_sources']['drop_score'] == 'json'
    assert provenance['config_source'] == str(path.resolve())
    assert provenance['factory_parameters'] == {'lang': 'ch', 'drop_score': threshold}


def test_defaults_and_explicit_override_have_distinct_sources(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text('{"drop_score":0.8}', encoding='utf-8')
    default = {}
    override = {}
    assert pipeline.process_configured('fixture', engine_factory=ScoreEngine, provenance=default)['pages'][0]['text'] == 'boundary\nbetween\nhigh'
    assert pipeline.process_configured('fixture', config_path=path, drop_score=0.5,
                            engine_factory=ScoreEngine, provenance=override)['pages'][0]['text'] == 'boundary\nbetween\nhigh'
    assert default['config_sha256'] is None
    assert default['value_sources']['drop_score'] == 'default'
    assert override['value_sources']['drop_score'] == 'explicit_override'
    assert override['declared_values'] == {'drop_score': 0.8}
    assert override['effective_values']['drop_score'] == 0.5


def test_file_changes_are_reloaded_and_not_hidden_by_previous_factory(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text('{"drop_score":0.5}', encoding='utf-8')
    old = {}
    new = {}
    first = pipeline.process_configured('fixture', config_path=path, engine_factory=ScoreEngine, provenance=old)
    path.write_text('{"drop_score":0.8}', encoding='utf-8')
    second = pipeline.process_configured('fixture', config_path=path, engine_factory=ScoreEngine, provenance=new)
    assert first['pages'][0]['text'] != second['pages'][0]['text']
    assert old['config_sha256'] != new['config_sha256']


@pytest.mark.parametrize('content', [
    '[]', 'null', '{"drop_score":true}', '{"drop_score":"0.5"}',
    '{"drop_score":NaN}', '{"drop_score":Infinity}', '{"drop_score":-0.1}',
    '{"drop_score":1.1}', '{"drop_score":0.5,"drop_score":0.8}',
    '{"drop_score":0.5,"module":"evil.py"}', '{"schema_version":2}',
    '{"schema_version":true}', '{"lang":"en"}', 'invalid JSON',
    '{"drop_score":0.5}' + ' ' * 16384,
])
def test_invalid_or_conflicting_json_rejected_before_engine(tmp_path, content):
    path = tmp_path / 'settings.json'
    path.write_text(content, encoding='utf-8')
    def forbidden_factory(**kwargs):
        pytest.fail('invalid configuration reached OCR constructor')
    with pytest.raises(ValueError):
        pipeline.process_configured('fixture', config_path=path, engine_factory=forbidden_factory)


@pytest.mark.parametrize('value', [True, '0.5', float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_override_rejected_before_engine(value):
    with pytest.raises(ValueError):
        ocr_client.create_configured_engine(drop_score=value, engine_factory=ScoreEngine)


@pytest.mark.parametrize('value', [10**500, -(10**500)])
def test_huge_integer_is_validation_error_not_overflow(value):
    with pytest.raises(ValueError):
        ocr_client.create_configured_engine(drop_score=value, engine_factory=ScoreEngine)
