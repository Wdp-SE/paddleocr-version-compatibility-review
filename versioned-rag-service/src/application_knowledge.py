"""Request-scoped original application evidence; never merged into the official SDK corpus."""
from collections import Counter
import hashlib
import math
import re
from src.paddleocr_compatibility import _files
from src.public_knowledge import tokens


def search_application(files, query, *, application_id, application_commit, top_k=5):
    if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', application_id)
            or not re.fullmatch(r'[0-9a-f]{40,64}', application_commit)
            or not isinstance(query, str) or not 1 <= len(query.strip()) <= 4000
            or type(top_k) is not int or not 1 <= top_k <= 10):
        raise ValueError('应用证据查询需要明确应用、完整提交和有界问题。')
    rows = _files(files)
    windows = []
    for row in rows:
        lines = row['content'].splitlines()
        # Preserve line boundaries for contracts, designs, tests and source evidence.
        for start in range(0, len(lines), 20):
            text = '\n'.join(lines[start:start + 20])
            windows.append({'path': row['path'], 'content': text, 'sha256': row['sha256'],
                            'line_start': start + 1, 'line_end': min(start + 20, len(lines)),
                            '_terms': Counter(tokens(row['path'] + ' ' + text))})
    terms = set(tokens(query)); count = len(windows)
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
    return {'results': hits[:top_k], 'retrieval_policy': 'application_snapshot_bm25',
            'knowledge_scope': 'submitted_application_commit', 'runtime_verified': False,
            'notice': '应用资料仅说明提交中的契约、代码或测试定义，不证明测试已执行；不作为官方 SDK 依据。'}
