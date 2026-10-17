import json
import subprocess
import pytest
from services.registered_application import load_registry, collect_snapshot


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    git(root, 'init'); git(root, 'config', 'user.email', 'demo@example.invalid')
    git(root, 'config', 'user.name', 'Original demonstration')
    (root / 'app').mkdir()
    files = {'caller.py': 'from paddleocr import PaddleOCR\n',
             'consumer.py': 'def consume(x): return x[0][1]\n',
             'contract.json': '{"required": ["text"]}',
             'test_contract.py': 'def test_contract(): pass\n',
             'requirements.txt': 'paddleocr==2.9.1\n'}
    for name, text in files.items(): (root / 'app' / name).write_text(text)
    git(root, 'add', '.'); git(root, 'commit', '-m', 'original baseline')
    before = git(root, 'rev-parse', 'HEAD')
    (root / 'app/requirements.txt').write_text('paddleocr==3.0.0\n')
    git(root, 'add', '.'); git(root, 'commit', '-m', 'candidate dependency')
    after = git(root, 'rev-parse', 'HEAD')
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'applications': [{'id': 'ocr', 'name': 'OCR',
        'repository': str(root), 'root': 'app', 'include': [''], 'exclude': [],
        'owner': 'original demo', 'source_identity': 'student_poc_application',
        'roles': {'entry': ['caller.py'], 'consumer': ['consumer.py'],
                  'contract': ['contract.json'], 'tests': ['test_contract.py'],
                  'dependency': ['requirements.txt']}}]}))
    return load_registry(registry)['ocr'], before, after, root


def test_dependency_only_change_still_collects_unchanged_consumer(repository):
    app, base, head, root = repository
    (root / 'app/consumer.py').write_text('uncommitted replacement')
    value = collect_snapshot(app, head, base_commit=base)
    assert value['head_commit'] == head and value['base_commit'] == base
    assert value['changed_paths'] == ['requirements.txt']
    assert 'x[0][1]' in next(r['content'] for r in value['files'] if r['path'] == 'consumer.py')
    assert value['coverage']['status'] == 'REGISTERED_SCOPE_COMPLETE'
    assert all(r['sha256'] for r in value['inventory'])


def test_missing_role_cannot_be_called_complete(repository):
    app, base, head, root = repository
    git(root, 'rm', 'app/caller.py'); git(root, 'commit', '-m', 'missing entry')
    value = collect_snapshot(app, 'HEAD', base_commit=base)
    assert value['coverage']['status'] == 'INCOMPLETE'
    assert 'entry' in value['coverage']['missing_roles']
    assert value['gaps']


@pytest.mark.parametrize('ref', ['--help', 'HEAD:app/caller.py', '../HEAD', 'HEAD\nHEAD', 'https://example.com'])
def test_ref_is_not_command_path_or_remote(repository, ref):
    with pytest.raises(ValueError): collect_snapshot(repository[0], ref)


def test_oversize_and_secret_files_are_not_silently_reviewed(repository):
    app, base, head, root = repository
    (root / 'app/large.py').write_text('x' * 50001)
    (root / 'app/.env').write_text('SECRET=never expose')
    git(root, 'add', '.'); git(root, 'commit', '-m', 'bounded files')
    value = collect_snapshot(app, 'HEAD')
    assert value['coverage']['status'] == 'INCOMPLETE'
    assert not any(r['path'] in ('.env', 'large.py') for r in value['files'])
    assert 'SECRET' not in json.dumps(value)


def test_registration_cannot_escape_repository(tmp_path):
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps({'applications': [{'id': 'bad', 'repository': '.', 'root': '../secret'}]}))
    with pytest.raises(ValueError): load_registry(path)


def test_symbolic_link_blobs_are_not_read_as_code(repository):
    app, base, head, root = repository
    blob = subprocess.check_output(['git', '-C', str(root), 'hash-object', '-w', '--stdin'], input=b'../../secret.txt').decode().strip()
    git(root, 'update-index', '--add', '--cacheinfo', '120000', blob, 'app/link.py')
    git(root, 'commit', '-m', 'link demonstration')
    value = collect_snapshot(app, 'HEAD')
    assert not any(row['path'] == 'link.py' for row in value['files'])
    assert value['coverage']['status'] == 'INCOMPLETE'


def test_public_default_registration_collects_real_original_app():
    value = collect_snapshot(load_registry()['document-ocr'], 'HEAD')
    assert value['coverage']['status'] == 'REGISTERED_SCOPE_COMPLETE'
    assert len(value['files']) == 12
    assert {r['path'] for r in value['files']} >= {'application/ocr_client.py', 'application/consumer.py', 'application/contract.json'}
