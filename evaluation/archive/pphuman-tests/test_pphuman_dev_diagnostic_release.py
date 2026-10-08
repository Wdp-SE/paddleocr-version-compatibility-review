import copy
import json
from pathlib import Path

from src import public_evaluation_release as release


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "versioned-rag-service" / "public_corpus_pphuman"
REPORT = ROOT / "evaluation" / "pphuman_release_candidate_v1" / "report.json"


def test_archived_dev_results_are_hidden_after_retrieval_code_changes():
    report = release.validate_pphuman_dev_diagnostic(CORPUS)
    assert report is None


def test_dev_report_with_wrong_corpus_fingerprint_is_not_published(monkeypatch):
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["input_fingerprints"]["chunks_sha256"] = "0" * 64
    monkeypatch.setattr(release, "_read", lambda _path: report)
    assert release.validate_pphuman_dev_diagnostic(CORPUS) is None


def test_dev_report_cannot_promote_itself_or_publish_impossible_metrics(monkeypatch):
    original = json.loads(REPORT.read_text(encoding="utf-8"))
    for mutate in (
        lambda row: row.update(default_promotion_eligible=True),
        lambda row: row["rag"]["bm25@5"].update(mean_required_source_recall=1.1),
        lambda row: row["input_fingerprints"].update({
            "code:versioned-rag-service/src/public_knowledge.py": "0" * 64,
        }),
    ):
        report = copy.deepcopy(original)
        mutate(report)
        monkeypatch.setattr(release, "_read", lambda _path: report)
        assert release.validate_pphuman_dev_diagnostic(CORPUS) is None
