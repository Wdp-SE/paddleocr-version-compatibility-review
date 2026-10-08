from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.public_knowledge import PublicKnowledgeIndex
from src.public_retrieval_runtime import PublicRetrievalRuntime


CORPUS = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"


def _config(tmp_path: Path, **overrides) -> Path:
    payload = json.loads((CORPUS / "public_retrieval_runtime.json").read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_runtime_is_exactly_the_locked_bm25_baseline(tmp_path):
    index = PublicKnowledgeIndex(CORPUS)
    runtime = PublicRetrievalRuntime(
        index, config_path=_config(tmp_path),
        sidecar_path=CORPUS / "figure_evidence_reviewed.json",
        inventory_path=CORPUS / "figure_evidence.json",
    )
    question = "PaddleOCR 的 OCR 结果包含哪些字段？"

    assert runtime.search(question, language="zh") == index.search(question, language="zh")
    assert runtime.last_retrieval_call_count == 1




@pytest.mark.parametrize("overrides", [
    {"schema_version": 2},
    {"default_policy": "unknown"},
    {"default_policy": "bm25_faceted_rrf", "allowed_policies": ["bm25"]},
    {"max_facets": 0},
])
def test_invalid_runtime_config_fails_closed(tmp_path, overrides):
    with pytest.raises(ValueError, match="runtime config"):
        PublicRetrievalRuntime(
            PublicKnowledgeIndex(CORPUS), config_path=_config(tmp_path, **overrides),
            sidecar_path=CORPUS / "figure_evidence_reviewed.json",
            inventory_path=CORPUS / "figure_evidence.json",
        )


def test_faceted_rrf_deduplicates_and_records_contributing_facets(tmp_path):
    index = PublicKnowledgeIndex(CORPUS)
    runtime = PublicRetrievalRuntime(
        index, config_path=_config(tmp_path),
        sidecar_path=CORPUS / "figure_evidence_reviewed.json",
        inventory_path=CORPUS / "figure_evidence.json",
    )
    hits = runtime.search(
        "PaddleOCR OCR 结果；识别参数和配置变更", top_k=10,
        version="current", language="zh", policy="bm25_faceted_rrf",
    )

    ids = [row["chunk_id"] for row in hits]
    assert len(ids) == len(set(ids))
    assert hits
    assert all(row["retrieval_policy"] == "bm25_faceted_rrf" for row in hits)
    assert runtime.last_retrieval_call_count >= 2


def test_single_fact_falls_back_to_one_bm25_query(tmp_path):
    index = PublicKnowledgeIndex(CORPUS)
    runtime = PublicRetrievalRuntime(
        index, config_path=_config(tmp_path),
        sidecar_path=CORPUS / "figure_evidence_reviewed.json",
        inventory_path=CORPUS / "figure_evidence.json",
    )
    query = "PaddleOCR v3.0.0 OCR"

    assert runtime.search(query, language="zh", policy="bm25_faceted_rrf") == index.search(query, language="zh")
    assert runtime.last_retrieval_call_count == 1


def test_current_corpus_does_not_claim_unreviewed_image_ocr_as_searchable():
    runtime = PublicRetrievalRuntime(PublicKnowledgeIndex(CORPUS))

    assert runtime.sidecar["chunks"] == []
    assert runtime._images == []
    assert "bm25_figure_ocr" not in runtime.config["allowed_policies"]
    with pytest.raises(ValueError, match="unsupported retrieval runtime policy"):
        runtime.search("接线图上的接口引脚", policy="bm25_figure_ocr")


def test_image_sidecar_for_another_manifest_is_rejected(tmp_path):
    sidecar = json.loads((CORPUS / "figure_evidence_reviewed.json").read_text(encoding="utf-8"))
    sidecar["corpus_manifest_sha256"] = "0" * 64
    sidecar_path = tmp_path / "tampered.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar does not match"):
        PublicRetrievalRuntime(
            PublicKnowledgeIndex(CORPUS), config_path=_config(tmp_path),
            sidecar_path=sidecar_path, inventory_path=CORPUS / "figure_evidence.json",
        )
