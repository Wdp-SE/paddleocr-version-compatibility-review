"""Opt-in paid-provider/real-HTTP smoke; outcomes are not answer accuracy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "change-review-agent"), str(ROOT / "demo-ui")]
from app.paddleocr_review import PaddleOCRReviewAgent
from config import DemoConfig
from services.public_knowledge_client import PublicKnowledgeClient


def run(base_url, *, generate, output_path=None):
    config = DemoConfig(rag_base_url=base_url, app_env="public_demo", allow_rag_query=generate,
                        runtime_root=ROOT / ".local-demo-temp/ocr-live-smoke", rag_retry_limit=0).validate()
    session = requests.Session()
    session.trust_env = False  # This client connects only to the explicit loopback URL.
    client = PublicKnowledgeClient(config.rag_base_url, session=session, retry_limit=0)
    workspace = client.workspace()
    if workspace.get("workspace_id") != "paddleocr":
        raise RuntimeError("The running backend is not the approved PaddleOCR workspace")
    reports = []
    agent = PaddleOCRReviewAgent(client)
    for name in ("removed_structure_api.py", "legacy_result_consumer.py", "compatible_basic.py"):
        file = ROOT / "examples/paddleocr_document_app" / name
        started = time.perf_counter()
        report = agent.analyze([{"path": name, "content": file.read_text(encoding="utf-8")}],
                               generate_advice=generate and name == "removed_structure_api.py")
        reports.append({"case": name, "latency_seconds": round(time.perf_counter() - started, 2), "report": report})
        print(name, report.get("status"), flush=True)
    answers = []
    if generate:
        for question in (
            "PaddleOCR 3.0 中的 PPStructure 应该迁移到哪个接口？",
            "PaddleOCR 3.0 的 OCR 结果 rec_texts 和 rec_scores 分别代表什么？",
        ):
            started = time.perf_counter()
            response = client.query_official(question, version="v3.0.0", language="zh", top_k=5)
            answers.append({"question": question, "latency_seconds": round(time.perf_counter() - started, 2),
                            "response": response})
            print("query", response.get("status"), response.get("generation", {}).get("returned_model"), flush=True)
    output = {"schema_version": 1, "project_id": "paddleocr", "assessment_type": "operational_smoke_not_accuracy",
              "workspace_fingerprint": workspace.get("corpus_fingerprint"), "generation_requested": generate,
              "application_reviews": reports, "answers": answers}
    path = Path(output_path) if output_path else Path(__file__).with_name("live_smoke_report.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", path.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--generate", action="store_true", help="Call the already configured paid provider")
    parser.add_argument("--output", help="Optional report path; preserve earlier operational runs")
    args = parser.parse_args()
    run(args.base_url, generate=args.generate, output_path=args.output)
