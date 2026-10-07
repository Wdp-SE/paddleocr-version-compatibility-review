import json
from pathlib import Path
from types import SimpleNamespace

from src.paddleocr_compatibility import review_compatibility


def review(files):
    root = Path(__file__).parents[1] / 'public_corpus_paddleocr'
    index = SimpleNamespace(manifest=json.loads((root/'corpus_manifest.json').read_text(encoding='utf8')),
                            chunks=json.loads((root/'chunks.json').read_text(encoding='utf8')))
    return review_compatibility(index, source_version='v2.9.1', target_version='v3.0.0', files=files)


def test_consumers_and_design_do_not_produce_missing_ocr_usage_noise():
    result = review([
        {'path': 'ocr_client.py', 'content': "from paddleocr import PaddleOCR\ndef recognize(path):\n    return PaddleOCR(lang='ch').ocr(path)\n"},
        {'path': 'consumer.py', 'content': "def output(rows):\n    return {'pages': rows}\n"},
        {'path': 'design.md', 'content': '输入扫描页，输出统一 JSON。'},
        {'path': 'contract.json', 'content': '{"required_line_fields":["text"]}'},
    ])
    assert not any(g['code'] in ('no_resolved_paddleocr_usage','unreviewed_configuration')
                   and g['application']['path'] != 'ocr_client.py' for g in result['gaps'])
    roles = {r['path']: r['role'] for r in result['files']}
    assert roles['design.md'] == 'design_document'
    assert roles['contract.json'] == 'output_contract'
    assert roles['consumer.py'] == 'application_code'


def test_unrelated_application_still_does_not_become_compatible():
    result = review([{'path': 'other.py', 'content': 'def foo():\n    return 1\n'}])
    assert result['status'] == 'needs_verification'
    assert any(g['code'] == 'no_resolved_paddleocr_usage' for g in result['gaps'])


def test_malformed_json_keeps_configuration_gap():
    result = review([{'path': 'contract.json', 'content': '{invalid'}])
    assert any(g['code'] == 'invalid_output_contract' for g in result['gaps'])


def test_structure_risk_requires_layout_regression_not_only_ocr_pages():
    result = review([{'path': 'layout.py', 'content': 'from paddleocr import PPStructure\nengine = PPStructure()\n'}])
    requirement = next(r for r in result['regression_requirements'] if r['rule_id'] == 'removed_ppstructure')
    assert '含表格与段落的版面页' in requirement['input_conditions']
    assert requirement['status'] == 'NOT_EXECUTED'
