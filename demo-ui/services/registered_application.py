"""Read immutable objects of server-registered local Git repositories. No checkout or execution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from services.paddleocr_application_inputs import parse_application_inputs

REGISTRY = Path(__file__).resolve().parents[2] / 'config/internal_applications.json'
EXTENSIONS = {'.py', '.json', '.md', '.txt', '.yaml', '.yml'}


def _path(value, *, empty=False):
    if empty and value == '': return value
    if (not isinstance(value, str) or not value or len(value) > 160 or '\\' in value or ':' in value
            or value.startswith('/') or any(ord(c) < 32 for c in value)
            or any(p in ('', '.', '..') for p in value.rstrip('/').split('/'))):
        raise ValueError('登记路径必须位于应用目录内。')
    return value


def load_registry(path=REGISTRY):
    """Only administrators edit this file. Repository paths are never browser input."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    result = {}
    for row in data['applications']:
        item = dict(row)
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', item['id']) or item['id'] in result:
            raise ValueError('应用登记标识不合法或重复。')
        item['root'] = _path(item['root']).rstrip('/')
        item['repository'] = str((path.parent / item['repository']).resolve())
        for key in ('include', 'exclude'):
            item[key] = [_path(p, empty=key == 'include') for p in item.get(key, [])]
        item['roles'] = {role: [_path(p) for p in paths] for role, paths in item['roles'].items()}
        item['registration_fingerprint'] = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result[item['id']] = item
    return result


def _git(app, *args, limit=2_000_000):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0',
               GIT_NO_REPLACE_OBJECTS='1', GIT_OPTIONAL_LOCKS='0')
    # Spool to disk, bounding memory even for a malformed or unexpectedly large tree.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        try:
            p = subprocess.run(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false',
                                '-c', 'safe.directory=' + app['repository'].replace('\\', '/'),
                                '-C', app['repository'], *args], stdout=output, stderr=error,
                               timeout=10, env=env, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError('登记仓库读取失败或超时；未获取远程资料。') from None
        if p.returncode or output.tell() > limit:
            raise ValueError('Git 对象不存在、读取失败或超出收集预算。')
        output.seek(0)
        return output.read(limit + 1)


def resolve_commit(app, ref):
    if (not isinstance(ref, str) or len(ref) > 160 or ref.startswith(('-', '/'))
            or '..' in ref or not re.fullmatch(r'[A-Za-z0-9_./@~-]+', ref)):
        raise ValueError('请输入本地提交 SHA 或分支名；不支持 URL、对象路径或命令。')
    sha = _git(app, 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}', limit=100).decode().strip()
    if not re.fullmatch(r'[0-9a-f]{40,64}', sha): raise ValueError('无法解析不可变提交。')
    return sha


def _selected(app, path):
    def match(pattern): return not pattern or path == pattern or (pattern.endswith('/') and path.startswith(pattern))
    return any(match(p) for p in app['include']) and not any(match(p) for p in app['exclude'])


def collect_snapshot(app, commit, *, base_commit=None):
    head = resolve_commit(app, commit)
    base = resolve_commit(app, base_commit) if base_commit else None
    prefix = app['root'] + '/'
    raw = _git(app, 'ls-tree', '-r', '-l', '-z', '--full-tree', head, '--', ':(literal)' + app['root'])
    files, inventory, skipped, gaps = [], [], [], []
    total = 0
    for record in sorted(raw.split(b'\0')):
        if not record: continue
        metadata, encoded = record.split(b'\t', 1)
        mode, kind, oid, size = metadata.decode('ascii').split()
        try: full_path = encoded.decode('utf-8'); relative = _path(full_path[len(prefix):])
        except (ValueError, UnicodeDecodeError):
            gaps.append({'code': 'unreadable_path', 'detail': '存在无法安全读取的路径。'}); continue
        if not full_path.startswith(prefix) or not _selected(app, relative): continue
        reason = None
        if mode not in ('100644', '100755') or kind != 'blob': reason = '链接或子模块不读取'
        elif any(p.startswith('.') or any(x in p.lower() for x in ('secret', 'credential', 'private_key')) for p in PurePosixPath(relative).parts): reason = '敏感文件名不读取'
        elif PurePosixPath(relative).suffix.lower() not in EXTENSIONS: reason = '非支持文本格式'
        elif int(size) > 50000 or total + int(size) > 200000 or len(files) >= 12: reason = '超出文件或文本预算'
        if reason:
            skipped.append({'path': relative, 'reason': reason}); continue
        content = _git(app, 'cat-file', 'blob', oid, limit=50000)
        try:
            text = content.decode('utf-8-sig')
            parse_application_inputs([{'path': relative, 'content': text}])
        except (UnicodeDecodeError, ValueError):
            skipped.append({'path': relative, 'reason': '非有效 UTF-8 或超过行数预算'}); continue
        files.append({'path': relative, 'content': text}); total += len(content)
        inventory.append({'path': relative, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'git_blob': oid, 'bytes': len(content)})
    if not files: raise ValueError('该提交的登记范围内没有可审查文本。')
    paths = {r['path'] for r in files}
    missing = {role: options for role, options in app['roles'].items() if not paths.intersection(options)}
    for role, options in missing.items():
        gaps.append({'code': 'missing_registered_role', 'detail': f'缺少 {role} 资料，需补充：' + ' / '.join(options)})
    for row in skipped:
        gaps.append({'code': 'uncollected_application_file', 'detail': row['path'] + '：' + row['reason']})
    changed = []
    if base:
        diff = _git(app, 'diff-tree', '--no-commit-id', '--name-only', '--no-renames', '-r', '-z', base, head, '--', ':(literal)' + app['root'])
        changed = sorted(p.decode('utf-8')[len(prefix):] for p in diff.split(b'\0') if p and p.decode('utf-8').startswith(prefix))
    return {'schema_version': 1, 'application_id': app['id'], 'application_name': app['name'],
            'owner': app['owner'], 'source_identity': app['source_identity'],
            'review_mode': 'candidate' if base else 'preflight', 'base_commit': base, 'head_commit': head,
            'registered_root': app['root'], 'files': parse_application_inputs(files), 'inventory': inventory,
            'registration_fingerprint': app['registration_fingerprint'],
            'changed_paths': changed, 'skipped': skipped, 'gaps': gaps,
            'coverage': {'status': 'INCOMPLETE' if gaps else 'REGISTERED_SCOPE_COMPLETE',
                         'missing_roles': missing, 'collected_count': len(files), 'byte_count': total,
                         'scope': '仅登记目录和角色；不证明完整依赖图或运行兼容', 'runtime_verified': False}}
