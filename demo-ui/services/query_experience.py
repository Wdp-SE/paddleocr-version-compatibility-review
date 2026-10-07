"""Session-local navigation metadata and business-oriented example copy."""
from time import monotonic


def cached_workspace(state, client, *, now=None, force=False):
    now = monotonic() if now is None else now
    cached = state.get('navigation_workspace_cache')
    if (not force and cached and cached['base_url'] == client.base_url
            and 0 <= now - cached['at'] < 30):
        return cached['workspace']
    # Failed refresh must never silently restore a previously ready backend.
    state.pop('navigation_workspace_cache', None)
    workspace = client.workspace()
    if workspace:
        state['navigation_workspace_cache'] = {
            'base_url': client.base_url, 'at': now, 'workspace': workspace,
        }
    return workspace


def interview_queries(workspace):
    queries = list(((workspace or {}).get('domain_profile') or {}).get('example_queries') or [])
    if (workspace or {}).get('workspace_id') != 'paddleocr':
        return queries
    replacements = {
        'PaddleOCR 3.0.0 的 OCR 结果中 rec_texts 和 rec_scores 分别表示什么？':
            'PaddleOCR 3.0.0 通用 OCR 的结果归一化读取中，rec_texts 与 rec_scores 有什么对应关系？text_rec_score_thresh 为什么会让部分识别结果未保留？',
        '仅凭官方资料，能否确认我们应用升级后下游 JSON 输出兼容？':
            '我们应用的 OCR 调用已改成 PaddleOCR 3.0.0 接口，是否就能确认下游 JSON 兼容？还需要核查哪些资料和测试？',
    }
    return [replacements.get(query, query) for query in queries]
