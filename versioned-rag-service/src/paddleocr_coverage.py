"""Task coverage, not a claim of complete enterprise knowledge."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.paddleocr_corpus import (
    PADDLEOCR_REPOSITORY, PADDLEOCR_VERSIONS, SERVICE_ROOT,
    safe_corpus_path, validate_paddleocr_manifest,
)

REQUIRED = {
    'installation': ('installation',),
    'ocr_contract': ('ocr_api',),
    'ocr_usage': ('ocr_pipeline', 'python_package'),
    'structure_contract': ('structure_api',),
    'structure_usage': ('structure_pipeline',),
    'configuration': ('inference_config', 'model_config', 'python_package'),
}
_IMAGES = re.compile(r'!\[[^\]]*\]\(([^\s)]+)[^)]*\)|<img\b[^>]*\bsrc=["\']([^"\']+)', re.I)


def audit_coverage(manifest: dict, trees: dict[str, dict],
                   application_manifest: dict | None = None, *, root: Path | None = None) -> dict:
    root = Path(root) if root is not None else SERVICE_ROOT / 'public_corpus_paddleocr'
    for tree in trees.values():
        if tree.get('truncated') is not False or not isinstance(tree.get('tree'), list):
            raise ValueError('truncated or incomplete Git tree')
    sources, media, requirements, gaps = [], [], [], []
    for row in manifest.get('sources', []):
        version, path = row['version'], row['path']
        if (row.get('repository') != PADDLEOCR_REPOSITORY or row.get('license') != 'Apache-2.0'
                or row.get('commit') != PADDLEOCR_VERSIONS.get(version, {}).get('commit')):
            raise ValueError('source identity is outside the fixed official project')
        tree_rows = {r['path']: r for r in trees.get(version, {}).get('tree', [])}
        if tree_rows.get(path, {}).get('sha') != row.get('git_blob_sha'):
            raise ValueError('source identity does not match complete fixed tree')
        sources.append({k: row[k] for k in ('source_id', 'version', 'path', 'sha256',
                                           'git_blob_sha', 'publisher', 'source_format', 'document_family')})
        if row['source_format'] == 'markdown':
            local = safe_corpus_path(root, row['local_path'])
            text = local.read_text(encoding='utf-8-sig')
            for line_no, line in enumerate(text.splitlines(), 1):
                for match in _IMAGES.finditer(line):
                    media.append({'source_id': row['source_id'], 'version': version,
                                  'line': line_no, 'reference': match[1] or match[2],
                                  'source_sha256': row['sha256'], 'status': 'unextracted',
                                  'license_status': 'not_individually_verified',
                                  'publisher': row['publisher']})
    for version in manifest.get('versions', {}):
        for name, families in REQUIRED.items():
            ids = [s['source_id'] for s in sources if s['version'] == version and s['document_family'] in families]
            req = {'version': version, 'family': name, 'source_ids': ids,
                   'status': 'available' if ids else 'missing'}
            requirements.append(req)
            if not ids:
                gaps.append({'code':'missing_task_family', 'version':version, 'family':name})
    if media:
        gaps.append({'code':'image_information_not_extracted', 'count':len(media)})
    formats = dict(Counter(s['source_format'] for s in sources))
    return {'schema_version': 1, 'task_id': 'paddleocr_document_application_upgrade',
            'coverage_claim': 'task_scoped_not_company_complete', 'sources': sources,
            'source_count': len(sources), 'formats': formats, 'required_families': requirements,
            'gaps': gaps, 'media': media,
            'application': {'source_type': 'developer_original_application',
                            'file_count': len((application_manifest or {}).get('files', []))}}


def validate_coverage(root: Path, coverage: dict) -> dict:
    root = Path(root)
    manifest = json.loads(safe_corpus_path(root, 'corpus_manifest.json').read_text(encoding='utf-8'))
    validate_paddleocr_manifest(root, manifest)
    trees = {v: json.loads(safe_corpus_path(root, f'provenance/tree-{v[1:]}.json').read_text(encoding='utf-8'))
             for v in manifest['versions']}
    expected = audit_coverage(manifest, trees, root=root)
    # Application documents are validated by their own manifest, never counted as official.
    expected['application'] = coverage.get('application')
    if coverage != expected:
        raise ValueError('coverage source identity, counts, requirements or media mismatch')
    for row in manifest['sources']:
        raw = safe_corpus_path(root, row['local_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('coverage source identity hash mismatch')
    return {'status': 'VERIFIED', 'source_count': expected['source_count'],
            'formats': expected['formats'], 'gaps': expected['gaps']}
