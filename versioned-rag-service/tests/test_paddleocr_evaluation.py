import copy
from pathlib import Path

from src import paddleocr_evaluation as release


CORPUS = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"


def test_only_current_source_derived_acceptance_can_be_published():
    report = release.load_paddleocr_acceptance(CORPUS)
    assert report is None  # v1 precedes the expanded 58-source corpus.


def test_stale_or_impossible_results_are_hidden(monkeypatch):
    original_read = release._read
    original = release._read(Path(__file__).resolve().parents[2]/'evaluation/paddleocr_compatibility_v1/report.json')
    assert original is not None
    for mutation in (
        lambda row: row.update(corpus_manifest_sha256="0" * 64),
        lambda row: row.update(compatibility_code_sha256="0" * 64),
        lambda row: row.update(fusion_code_sha256="0" * 64),
        lambda row: row.update(answer_accuracy=1.0),
        lambda row: row["retrieval"]["bm25"].update(evidence_span_hit_count=99),
        lambda row: row["compatibility"][0].update(runtime_verified=True),
    ):
        report = copy.deepcopy(original)
        mutation(report)
        monkeypatch.setattr(release, "_read", lambda path: report if path.name == "report.json" else original_read(path))
        assert release.load_paddleocr_acceptance(CORPUS) is None
