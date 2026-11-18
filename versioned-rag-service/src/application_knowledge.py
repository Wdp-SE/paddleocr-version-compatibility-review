"""Request-scoped original application evidence; never merged into the official SDK corpus."""
import ast
from collections import Counter
import hashlib
import math
import re
from src.paddleocr_compatibility import _files
from src.public_knowledge import tokens
from src.paddleocr_application_roles import application_role
from src.internal_workflows import WORKFLOWS


def resolve_entrypoint(registration, snapshot, entrypoint):
    """Resolve an explicitly selected entry and exact dependency pin from one Git snapshot."""
    manifest = registration.get('entrypoints', {}).get(entrypoint)
    context = {'entrypoint': entrypoint, 'declared_version': None, 'runtime_verified': False}
    if not manifest:
        return {**context, 'gap': '请选择已登记的应用入口。'}
    path, _, symbol = entrypoint.partition('::')
    files = {row['path']: row['content'] for row in snapshot['files']}
    try:
        nodes = ast.parse(files.get(path, '')).body
    except (SyntaxError, ValueError, RecursionError):
        nodes = []
    definitions = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == symbol]
    if len(definitions) != 1:
        return {**context, 'gap': '本次提交未提供唯一可定位的入口定义。'}
    lines = files.get(manifest, '').splitlines()
    matches = [(number, line) for number, line in enumerate(lines, 1)
               if re.match(r'^\s*paddleocr(?=\s|[<=>!~;\[@]|$)', line, re.IGNORECASE)]
    if len(matches) != 1 or any(line.lstrip().startswith('-') for line in lines):
        return {**context, 'gap': '入口依赖文件缺失，或存在多项声明、外部包含及安装选项。'}
    number, line = matches[0]
    pin = re.fullmatch(r'\s*paddleocr\s*==\s*v?(\d+\.\d+\.\d+)\s*(?:#.*)?', line, re.IGNORECASE)
    version = 'v' + pin[1] if pin else None
    if version not in WORKFLOWS['upgrade_review']:
        return {**context, 'gap': '入口依赖没有明确固定到已收录的 PaddleOCR 版本。'}
    return {**context, 'declared_version': version,
            'entry_source': {'path': path, 'line': definitions[0].lineno},
            'dependency_source': {'path': manifest, 'line': number}}


def _units(row):
    lines = row['content'].splitlines()
    spans = []
    if row['path'].endswith('.py'):
        from src.paddleocr_compatibility import _tree
        try:
            tree = _tree(row['content'])
            spans = [(n.lineno-1, n.end_lineno) for n in tree.body if hasattr(n, 'end_lineno')]
        except (SyntaxError, ValueError):
            pass
    if not spans:
        spans = [(0, len(lines))]
    for lo, hi in spans:
        for start in range(lo, hi, 50):
            end = min(start+60, hi)
            text = '\n'.join(lines[start:end])
            if text.strip():
                yield start+1, end, text
            if end == hi:
                break


def _intent(query):
    q = query.casefold()
    roles, aliases = [], []
    page_question = any(t in q for t in ('空页', '页号', '页码', '页序', '第几页', '没字', '空白页', 'page_index'))
    if page_question or any(t in q for t in ('契约', '字段', 'contract', 'schema', 'page_policy', 'output_fields')):
        roles.append('output_contract')
        aliases.extend(['required_line_fields', 'output_fields', 'page_policy', 'page_fields'])
    if any(t in q for t in ('测试', '验收', '断言', '回归', 'assert', 'test_')):
        roles.append('application_test')
        aliases.extend(['test', 'assert'])
    if page_question:
        roles.append('application_code')
        aliases.extend(['page_index', 'page_count', 'preserve_empty_pages', 'to_json'])
    return roles, aliases


def search_application(files, query, *, application_id, application_commit, top_k=5):
    if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', application_id)
            or not re.fullmatch(r'[0-9a-f]{40,64}', application_commit)
            or not isinstance(query, str) or not 1 <= len(query.strip()) <= 4000
            or type(top_k) is not int or not 1 <= top_k <= 10):
        raise ValueError('应用证据查询需要明确应用、完整提交和有界问题。')
    rows = _files(files)
    windows = []
    for row in rows:
        role = application_role(row['path'], row['content'])
        # Preserve small JSON documents and Python top-level structures, never stitch excerpts.
        for start, end, text in _units(row):
            windows.append({'path': row['path'], 'content': text, 'sha256': row['sha256'],
                            'line_start': start, 'line_end': end, 'document_role': role,
                            '_terms': Counter(tokens(row['path'] + ' ' + text))})
    requested_roles, aliases = _intent(query)
    terms = set(tokens(query+' '+' '.join(aliases))); count = len(windows)
    df = Counter(t for window in windows for t in window['_terms'])
    avg = sum(sum(w['_terms'].values()) for w in windows) / max(count, 1)
    hits = []
    for row in windows:
        bag = row.pop('_terms'); length = sum(bag.values())
        score = sum(math.log(1 + (count - df[t] + .5) / (df[t] + .5)) *
                    bag[t] * 2.2 / (bag[t] + 1.2 * (.25 + .75 * length / max(avg, 1)))
                    for t in terms if bag[t])
        if score > 0:
            hits.append({**row, 'score': round(score, 6), 'application_id': application_id,
                         'application_commit': application_commit, 'source_identity': 'submitted_application',
                         'evidence_id': 'app-' + hashlib.sha256(f"{application_commit}:{row['path']}:{row['line_start']}".encode()).hexdigest()[:24]})
    hits.sort(key=lambda r: (-r['score'], r['path'], r['line_start']))
    selected = []
    # A multi-part contract/test question needs both evidence types, not repeated noisy code.
    for role in requested_roles:
        candidate = next((r for r in hits if r['document_role']==role and r not in selected), None)
        if candidate and len(selected)<top_k:
            selected.append(candidate)
    selected.extend(r for r in hits if r not in selected)
    return {'results': selected[:top_k], 'retrieval_policy': 'application_snapshot_bm25',
            'evidence_strategy': 'structure_units_and_requested_roles',
            'knowledge_scope': 'submitted_application_commit', 'runtime_verified': False,
            'notice': '应用资料仅说明提交中的契约、代码或测试定义，不证明测试已执行；不作为官方 SDK 依据。'}
