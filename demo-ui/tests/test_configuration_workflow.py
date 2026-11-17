import copy
import ast
import hashlib
import json
from pathlib import Path
import sys
import pytest
from services.configuration_workflow import configuration_demo_inputs, attach_configuration_snapshots
from services.paddleocr_application_inputs import load_original_application


def test_demo_has_real_configuration_delta_and_keeps_same_application_files():
    before, after = configuration_demo_inputs()
    assert [f['path'] for f in before] == [f['path'] for f in after]
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 1
    assert 'drop_score=0.5' in changed[0][0]['content']
    assert 'drop_score=0.8' in changed[0][1]['content']


def test_json_configuration_demo_preserves_original_contract_and_declares_added_file():
    before,after=configuration_demo_inputs(style='json')
    from services.paddleocr_application_inputs import parse_application_inputs
    assert parse_application_inputs(before) and parse_application_inputs(after)
    from services.paddleocr_application_inputs import load_original_application
    original={row['path'] for row in load_original_application()}
    assert original-{row['path'] for row in before}=={'application/upgrade.md'}
    old=next(f for f in before if f['path']=='application/ocr_settings.json')
    new=next(f for f in after if f['path']==old['path'])
    assert '0.5' in old['content'] and '0.8' in new['content']
    assert next(f for f in before if f['path']=='application/contract.json')==next(f for f in after if f['path']=='application/contract.json')
    client=next(f for f in after if f['path']=='application/ocr_client.py')['content']
    assert json.loads(old['content']) == {'schema_version': 1, 'drop_score': 0.5}
    assert json.loads(new['content']) == {'schema_version': 1, 'drop_score': 0.8}
    assert 'json.load' in client and "drop_score=settings['drop_score']" in client


@pytest.mark.parametrize('style', ['direct', 'json'])
def test_current_runtime_factory_can_generate_actionable_static_fixture_without_claiming_execution(style):
    root = Path(__file__).resolve().parents[2]
    runtime_path = root/'examples/paddleocr_document_app/application/ocr_client.py'
    runtime_source = runtime_path.read_bytes()
    before, after = configuration_demo_inputs(style=style)
    sys.path.insert(0, str(root/'versioned-rag-service'))
    from src.paddleocr_configuration import review_configuration
    result = review_configuration(before, after, dependency_version='v2.9.1')
    assert [(row['before_value'], row['after_value']) for row in result['changes']] == [(0.5, 0.8)]
    assert result['gaps'] == []
    assert result['runtime_verified'] is False
    original = {row['path']: row for row in load_original_application()}
    for snapshot in [before, after]:
        client = next(row['content'] for row in snapshot if row['path'] == 'application/ocr_client.py')
        provenance_text = ast.get_docstring(ast.parse(client))
        assert provenance_text and provenance_text.startswith('{'), 'generated fixture must carry explicit provenance'
        provenance = json.loads(provenance_text)
        assert provenance['source_type'] == 'developer_original_static_fixture'
        assert provenance['base_application_sha256'] == hashlib.sha256(runtime_source).hexdigest()
        assert provenance['runtime_verified'] is False
        assert all(row == original[row['path']] for row in snapshot
                   if row['path'] not in ('application/ocr_client.py', 'application/ocr_settings.json'))
    assert runtime_path.read_bytes() == runtime_source


def test_registered_config_requires_same_application_and_explicit_base_commit():
    snapshot = {'application_id': 'a', 'registered_root': 'root', 'source_identity': 'student_poc_application',
                'head_commit': 'a'*40, 'gaps': [], 'coverage': {'status': 'REGISTERED_SCOPE_COMPLETE'}, 'files': []}
    result = {'workflow_type': 'configuration_change', 'compatibility_report': {'gaps': [], 'findings': []}}
    with pytest.raises(ValueError): attach_configuration_snapshots(result, snapshot, None)
    candidate = {**snapshot, 'head_commit': 'b'*40, 'base_commit': 'a'*40}
    value = attach_configuration_snapshots(copy.deepcopy(result), candidate, snapshot)
    assert value['baseline_application_snapshot']['head_commit'] == 'a'*40
    with pytest.raises(ValueError):
        attach_configuration_snapshots(result, {**candidate, 'application_id': 'other'}, snapshot)
