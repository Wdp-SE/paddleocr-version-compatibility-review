"""Reproduce source-derived acceptance; do not call this a blind accuracy test."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "versioned-rag-service"))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_compatibility import review_compatibility
from src.retrieval_fusion import fuse_ranked_hits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    evaluation = Path(__file__).resolve().parent
    cases = json.loads((evaluation / "cases.json").read_text(encoding="utf-8"))
    corpus = ROOT / "versioned-rag-service" / "public_corpus_paddleocr"
    index = PublicKnowledgeIndex(corpus)
    top_k = cases["top_k"]
    results = {}
    for policy in ("bm25", "bm25_fields", "bm25_fields_rrf"):
        outcomes = []
        for case in cases["retrieval_cases"]:
            started = time.perf_counter()
            kwargs = {"version": case["version"], "language": "zh"}
            if policy == "bm25_fields_rrf":
                rankings = [index.search(case["query"], top_k=20, policy=p, **kwargs)
                            for p in ("bm25", "bm25_fields")]
                rows = fuse_ranked_hits(rankings, top_k=top_k, rrf_k=60)
            else:
                rows = index.search(case["query"], top_k=top_k, policy=policy, **kwargs)
            matched = [row for row in rows if row["document_path"] == case["path"]
                       and all(token in row["content"] for token in case["tokens"])]
            outcomes.append({
                "id": case["id"], "version": case["version"], "matched": bool(matched),
                "matched_chunk_ids": [row["chunk_id"] for row in matched],
                "returned_paths": [row["document_path"] for row in rows],
                "wrong_version_count": sum(row["version"] != case["version"] for row in rows),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            })
        results[policy] = {"count": len(outcomes), "evidence_span_hit_count": sum(row["matched"] for row in outcomes),
                           "wrong_version_count": sum(row["wrong_version_count"] for row in outcomes), "cases": outcomes}
    compatibility = []
    for case in cases["compatibility_cases"]:
        app = ROOT / "examples" / "paddleocr_document_app" / case["file"]
        report = review_compatibility(index, source_version="v2.9.1", target_version="v3.0.0",
                                     files=[{"path": app.name, "content": app.read_text(encoding="utf-8")}])
        rules = [row["rule_id"] for row in report["findings"]]
        compatibility.append({"id": case["id"], "application_sha256": digest(app), "status": report["status"],
                              "rules": rules, "passed": report["status"] == case["expected_status"]
                              and case["expected_rule"] in rules,
                              "runtime_verified": report["runtime_verified"]})
    output = {"schema_version": 1, "dataset_id": cases["dataset_id"], "scope": cases["purpose"],
              "project_id": "paddleocr", "assessment_type": "source_derived_acceptance",
              "cases_sha256": digest(evaluation / "cases.json"), "corpus_manifest_sha256": digest(corpus / "corpus_manifest.json"),
              "chunks_sha256": digest(corpus / "chunks.json"),
              "runner_sha256": digest(evaluation / "run_evaluation.py"),
              "retrieval_code_sha256": digest(ROOT / "versioned-rag-service" / "src" / "public_knowledge.py"),
              "fusion_code_sha256": digest(ROOT / "versioned-rag-service" / "src" / "retrieval_fusion.py"),
              "compatibility_code_sha256": digest(ROOT / "versioned-rag-service" / "src" / "paddleocr_compatibility.py"),
              "top_k": top_k, "retrieval": results, "compatibility": compatibility,
              "answer_accuracy": None, "independent_business_accuracy": None,
              "selection_decision": "保留 BM25 基线；小样本来源派生验收不能证明业务准确率，不自动晋升候选策略。"}
    (evaluation / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"retrieval": {p: {k: v for k, v in r.items() if k != "cases"} for p, r in results.items()},
                      "compatibility": compatibility, "scope": cases["purpose"]}, ensure_ascii=False, indent=2))
    return all(case["passed"] for case in compatibility)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
