import copy
from pathlib import Path
from src import paddleocr_evaluation as release

CORPUS=Path(__file__).parents[1]/'public_corpus_paddleocr'


def test_stale_quality_comparison_is_hidden(monkeypatch):
    monkeypatch.setattr(release,'_read',lambda path:{'dataset_id':'paddleocr-quality-v2','chunks_sha256':'stale'})
    assert release.load_quality_comparison(CORPUS) is None


def test_runtime_check_does_not_imply_uploaded_app_was_executed(monkeypatch):
    monkeypatch.setattr(release,'_read',lambda path:{'sample_regression_passed':True,'adapter_sha256':'stale'})
    assert release.load_upgrade_demo() is None


def test_current_comparison_and_demo_publish_without_business_accuracy_claims():
    report=release.load_quality_comparison(CORPUS)
    assert report is None  # Preserve v2 as history; do not refresh its hashes.
    runtime=release.load_upgrade_demo()
    if runtime is not None:assert runtime['uploaded_application_runtime_verified'] is False


def test_independent_results_do_not_survive_comparison_drift(monkeypatch):
    monkeypatch.setattr(release,'_read',lambda path:{'comparison_sha256':'stale'})
    assert release.load_independent_probes(CORPUS) is None


def test_inflated_quality_metrics_are_not_published(monkeypatch):
    original_read=release._read
    report=copy.deepcopy(release._read(Path(__file__).parents[2]/'evaluation/paddleocr_quality_v2/report.json'))
    assert report is not None
    report['strategies']['bm25']['holdout']['complete_rate']=1.0
    monkeypatch.setattr(release,'_read',lambda path:report if path.name=='report.json' else original_read(path))
    assert release.load_quality_comparison(CORPUS) is None
