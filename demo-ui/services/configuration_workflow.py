"""Configuration branch adapters; shared provenance and disposition stay reusable."""
import copy
import hashlib
import json
import sys
from pathlib import Path

from services.application_review_context import attach_application_context
from services.paddleocr_application_inputs import load_original_application


def configuration_demo_inputs(*,style='direct'):
    """Static original-app fixtures with provenance; no Git or runtime claims."""
    if style not in ('direct','json'):raise ValueError('Unsupported configuration demonstration')
    files = load_original_application()
    if style=='json':
        # Same-version configuration review does not require the dependency
        # upgrade narrative. Keep the accompanying app source, contracts and
        # tests, making room for declared config within the 12-file budget.
        files=[row for row in files if row['path']!='application/upgrade.md']
    before, after = copy.deepcopy(files), copy.deepcopy(files)
    for rows, threshold in [(before, '0.5'), (after, '0.8')]:
        for row in rows:
            if row['path'] == 'application/ocr_client.py':
                provenance = {'source_type': 'developer_original_static_fixture',
                    'derived_from': row['path'],
                    'base_application_sha256': hashlib.sha256(row['content'].encode('utf-8')).hexdigest(),
                    'runtime_verified': False,
                    'scope': '原创应用静态声明对照夹具；此调用层经过简化，未执行。真实 JSON 加载与 OCR 记录独立保存。'}
                source = '"""' + json.dumps(provenance, ensure_ascii=False, indent=2) + '"""\n'
                source += 'from paddleocr import PaddleOCR\n'
                if style=='json':
                    source += 'import json\n\n\ndef create_engine():\n'
                    source += '    settings = json.load(open("application/ocr_settings.json", encoding="utf-8"))\n'
                    value = "settings['drop_score']"
                else:
                    source += '\n\ndef create_engine():\n'
                    value = threshold
                source += f"    return PaddleOCR(lang='ch', drop_score={value})\n"
                source += '\n\ndef recognize(path):\n    engine = create_engine()\n    return engine.ocr(path)\n'
                row['content'] = source
        if style=='json':rows.append({'path':'application/ocr_settings.json','content':json.dumps({'schema_version':1,'drop_score':float(threshold)},indent=2)})
    return before, after


def attach_configuration_snapshots(result, snapshot, base_snapshot):
    if (not base_snapshot or snapshot.get('base_commit') != base_snapshot.get('head_commit')
            or any(snapshot.get(k) != base_snapshot.get(k)
                   for k in ('application_id', 'registered_root', 'source_identity'))):
        raise ValueError('配置审查必须比较同一登记应用的基准与候选提交。')
    result = copy.deepcopy(result)
    result['baseline_application_snapshot'] = {k: v for k, v in base_snapshot.items() if k != 'files'}
    result['compatibility_report']['gaps'].extend(copy.deepcopy(base_snapshot.get('gaps', [])))
    return attach_application_context(result, snapshot)


def run_configuration_review(client, before_files, after_files, *, snapshot=None, base_snapshot=None, graph_runtime=None):
    root = str(Path(__file__).resolve().parents[2] / 'change-review-agent')
    if root in sys.path: sys.path.remove(root)
    sys.path.insert(0, root)
    from app.paddleocr_configuration_review import review_configuration_change
    options = {'graph_runtime': graph_runtime} if graph_runtime is not None else {}
    result = review_configuration_change(client, before_files, after_files, **options)
    result['application_context'] = {'application_name': '扫描文档处理应用',
        'business_scope': 'internal_document_processing', 'source_identity': 'student_poc_application',
        'change_reason': '同一依赖版本下审查 OCR 置信度过滤阈值变更'}
    if snapshot:
        return attach_configuration_snapshots(result, snapshot, base_snapshot)
    result['input_mode'] = 'original_application_variants'
    return result


def configuration_identity(before, after, workspace):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root/'change-review-agent/app/paddleocr_configuration_review.py',
             root/'demo-ui/services/paddleocr_disposition.py', root/'demo-ui/services/application_review_context.py']
    return hashlib.sha256(json.dumps({'workflow_type': 'configuration_change', 'before': before, 'after': after,
        'workspace': {k: (workspace or {}).get(k) for k in ('corpus_fingerprint', 'service_code_fingerprint', 'retrieval_config_fingerprint')},
        'code': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
