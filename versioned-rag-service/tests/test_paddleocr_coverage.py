import copy
import json
from pathlib import Path

import pytest


def test_truncated_tree_cannot_be_a_complete_coverage_audit():
    from src.paddleocr_coverage import audit_coverage
    with pytest.raises(ValueError, match='truncated'):
        audit_coverage({'sources': []}, {'v3.0.0': {'truncated': True, 'tree': []}})


def test_real_coverage_binds_sources_and_records_unextracted_media():
    from src.paddleocr_coverage import audit_coverage, validate_coverage
    root = Path(__file__).resolve().parents[1] / 'public_corpus_paddleocr'
    manifest = json.loads((root / 'corpus_manifest.json').read_text(encoding='utf-8'))
    trees = {v: json.loads((root/'provenance'/f'tree-{v[1:]}.json').read_text(encoding='utf-8'))
             for v in manifest['versions']}
    report = audit_coverage(manifest, trees)
    assert report['source_count'] == len(manifest['sources'])
    assert report['formats'].get('docx', 0) == 0
    assert report['formats'].get('xlsx', 0) == 0
    assert validate_coverage(root, report)['status'] == 'VERIFIED'
    forged = copy.deepcopy(report)
    forged['sources'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        validate_coverage(root, forged)
    assert all(row['publisher'] == 'PaddlePaddle' for row in report['sources'])
    assert report['coverage_claim'] == 'task_scoped_not_company_complete'


def test_original_materials_do_not_inflate_official_coverage():
    from src.paddleocr_coverage import audit_coverage
    report = audit_coverage({'sources': [], 'versions': {}}, {}, {'files': [{'path':'design.md'}]})
    assert report['source_count'] == 0
    assert report['application']['source_type'] == 'developer_original_application'
    assert report['application']['file_count'] == 1
