import importlib.util
import json
from pathlib import Path
import subprocess


def test_read_only_cli_binds_git_snapshot_and_does_not_approve(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/review_registered_upgrade.py'
    spec=importlib.util.spec_from_file_location('upgrade_cli',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output=tmp_path/'report.json'
    code=module.main(['--head','HEAD','--output',str(output),'--advisory'])
    report=json.loads(output.read_text(encoding='utf8'))
    assert code==0 and report['gate']=='ADVISORY_NOT_APPROVAL'
    assert len(report['application_commit'])==40 and report['runtime_verified'] is False
    assert report['review']['summary']['risk_count']>0
    assert report['input_scope']['collected_count']==12
    from services.registered_application import collect_snapshot, load_registry
    from src.application_knowledge import resolve_entrypoint
    registration = load_registry()['document-ocr']
    snapshot = collect_snapshot(registration, report['application_commit'])
    resolved = resolve_entrypoint(registration, snapshot, 'application/pipeline.py::process')
    assert resolved['declared_version'] == 'v2.9.1'
    assert resolved['dependency_source']['path'] == 'application/requirements-v2.txt'
    assert resolved['runtime_verified'] is False


def test_review_compares_two_real_git_snapshots_including_unchanged_consumer(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    path = root / 'versioned-rag-service/scripts/review_registered_upgrade.py'
    spec = importlib.util.spec_from_file_location('compare_upgrade_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from services import registered_application as registry
    from services.interview_examples import demo_application

    repo = tmp_path / 'original-application'
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT,
                                       encoding='utf-8', errors='replace').strip()

    def commit(case):
        for row in demo_application(case):
            file = repo / 'ocr' / row['path']
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(row['content'], encoding='utf-8')
        git('add', '--', 'ocr')
        git('-c', 'user.name=OCR Fixture', '-c', 'user.email=ocr@example.invalid',
            '-c', 'commit.gpgSign=false', 'commit', '-m', case)
        return git('rev-parse', 'HEAD')

    git('init')
    before = commit('升级风险：旧结果消费')
    after = commit('修改后复查：字段适配')
    app = {**registry.load_registry()['document-ocr'], 'repository': str(repo), 'root': 'ocr'}
    monkeypatch.setattr(registry, 'load_registry', lambda: {'document-ocr': app})
    report_path = tmp_path / 'comparison.json'
    assert cli.main(['--base', before, '--head', after, '--advisory', '--output', str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert (report['base_application_commit'], report['application_commit']) == (before, after)
    assert report['baseline_review']['summary']['risk_count'] > 0
    assert report['review']['summary']['risk_count'] == 0
    assert report['changed_paths'] == ['application/normalizer.py']
    assert any(row['path'] == 'application/consumer.py' for row in report['inventory'])
    assert report['runtime_verified'] is False
