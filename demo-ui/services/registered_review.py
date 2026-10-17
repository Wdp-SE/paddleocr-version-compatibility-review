"""Orchestrate fixed-application snapshots through the existing Agent and knowledge service."""
from services.application_review_context import attach_application_context, finalize_application_report
from services.review_comparison import capture_baseline, compare_reviews
import hashlib
import json


def validate_application_evidence(value, snapshot):
    if not isinstance(value, dict) or not isinstance(value.get('results'), list) or len(value['results']) > 10:
        raise ValueError('应用资料检索响应无效。')
    files = {row['path']: row['content'] for row in snapshot['files']}
    for row in value['results']:
        content = files.get(row.get('path'))
        start, end = row.get('line_start'), row.get('line_end')
        if (content is None or type(start) is not int or type(end) is not int
                or not 1 <= start <= end <= len(content.splitlines())
                or row.get('application_id') != snapshot['application_id']
                or row.get('application_commit') != snapshot['head_commit']
                or row.get('source_identity') != 'submitted_application'
                or row.get('sha256') != hashlib.sha256(content.encode()).hexdigest()
                or row.get('content') != '\n'.join(content.splitlines()[start-1:end])):
            raise ValueError('应用证据与本次提交内容不一致。')
    return value


def run_registered_review(client, snapshot, *, analyze, context, base_snapshot=None,
                          source_version, target_version, generate_advice=False, **options):
    def review(current, *, advice, resume=None):
        result = analyze(client, current['files'], source_version=source_version, target_version=target_version,
                         generate_advice=advice, **{**options, 'resume': resume})
        result['application_context'] = context
        return attach_application_context(result, current)
    baseline = review(base_snapshot, advice=False) if base_snapshot else None
    result = review(snapshot, advice=generate_advice, resume=options.get('resume'))
    try:
        result['application_evidence'] = validate_application_evidence(client.application_search(snapshot['files'],
            'required_line_fields output_fields text confidence page_index schema_version pages 测试 契约',
            application_id=snapshot['application_id'], application_commit=snapshot['head_commit']), snapshot)
    except Exception:
        result['application_evidence'] = {'results': [], 'status': 'UNAVAILABLE', 'runtime_verified': False}
        result['compatibility_report']['gaps'].append({'code': 'application_evidence_unavailable',
            'detail': '当前提交的应用资料检索未完成；请直接核对已提交的契约、设计和测试文件。'})
        # Bind the additional gap too; never leave a report fingerprint from before enrichment.
        finalize_application_report(result)
    if baseline:
        result['commit_comparison'] = compare_reviews(
            capture_baseline(baseline, corpus_fingerprint=baseline.get('corpus_fingerprint')),
            result, corpus_fingerprint=result.get('corpus_fingerprint'))
    return result
