import hashlib
import subprocess

from src.build_identity import content_fingerprint, git_revision


def test_git_revision_requires_a_valid_commit_and_clean_worktree(monkeypatch, tmp_path):
    sha = "a" * 40
    replies = iter([
        subprocess.CompletedProcess([], 0, stdout=sha + "\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    ])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(replies))

    assert git_revision(tmp_path) == sha


def test_git_revision_uses_valid_render_deploy_commit(monkeypatch, tmp_path):
    sha = "c" * 40
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_GIT_COMMIT", sha)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("git should not be needed")))

    assert git_revision(tmp_path) == sha


def test_git_revision_ignores_malformed_render_deploy_commit(monkeypatch, tmp_path):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_GIT_COMMIT", "latest")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr=""))

    assert git_revision(tmp_path) is None


def test_git_revision_reports_unknown_for_absent_or_dirty_checkout(monkeypatch, tmp_path):
    sha = "b" * 40
    replies = iter([
        subprocess.CompletedProcess([], 0, stdout=sha + "\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout=" M app.py\n", stderr=""),
        FileNotFoundError("git"),
    ])

    def run(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(subprocess, "run", run)

    assert git_revision(tmp_path) is None
    assert git_revision(tmp_path) is None


def test_git_revision_reports_unknown_on_timeout(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", timeout)

    assert git_revision(tmp_path, timeout_seconds=0.01) is None


def test_content_fingerprint_changes_with_file_content(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("first", encoding="utf-8")
    first = content_fingerprint({"manifest_sha256": path})
    path.write_text("second", encoding="utf-8")
    second = content_fingerprint({"manifest_sha256": path})

    assert first["manifest_sha256"] == hashlib.sha256(b"first").hexdigest()
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


def test_content_fingerprint_marks_missing_inputs_unknown(tmp_path):
    result = content_fingerprint({"chunks_sha256": tmp_path / "missing.json"})

    assert result == {"chunks_sha256": None, "fingerprint_sha256": "unknown"}


def test_review_identity_changes_when_agent_or_orchestration_changes(tmp_path):
    from src.build_identity import review_execution_fingerprint
    for folder,name in (('versioned-rag-service/src','api.py'),('change-review-agent/app','review.py'),
        ('change-review-agent/config','policy.json'),('demo-ui/services','registered_review.py')):
        directory=tmp_path/folder;directory.mkdir(parents=True)
        (directory/name).write_text('{}')
    before=review_execution_fingerprint(tmp_path)
    (tmp_path/'change-review-agent/app/review.py').write_text('changed')
    assert before!=review_execution_fingerprint(tmp_path)


def test_project_workspace_does_not_publish_legacy_domain_evaluation_fingerprint(tmp_path):
    from src.build_identity import public_build_identity

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_manifest.json").write_text(
        '{"project_id":"industrial-inspection"}', encoding="utf-8",
    )
    config = corpus / "public_retrieval_runtime.json"
    config.write_text("{}", encoding="utf-8")

    result = public_build_identity(
        repo_root=tmp_path,
        corpus_root=corpus,
        retrieval_config_path=config,
    )

    assert result["evaluation_fingerprint"] == "pending_project_evaluation"


def test_pphuman_build_identity_never_reuses_another_domain_evaluation(tmp_path):
    from src.build_identity import public_build_identity

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_manifest.json").write_text(
        '{"workspace_id":"pphuman"}', encoding="utf-8",
    )
    config = corpus / "public_retrieval_runtime.json"
    config.write_text("{}", encoding="utf-8")

    result = public_build_identity(
        repo_root=tmp_path,
        corpus_root=corpus,
        retrieval_config_path=config,
    )

    assert result["evaluation_fingerprint"] == "pending_pphuman_evaluation"
