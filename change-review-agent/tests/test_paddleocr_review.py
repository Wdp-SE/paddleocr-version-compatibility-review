from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.paddleocr_review import COMMITS, PaddleOCRReviewAgent, PROFILE_PATH
from app.domain_profile import load_change_profile


FILES = [{"path": "app/scan.py", "content": "from paddleocr import PPStructure\nengine = PPStructure()\n"}]


def source(version):
    path = "paddleocr.py" if version == "v2.9.1" else "paddleocr/__init__.py"
    return {"source_id": f"ocr-{version}", "document_id": f"{version}:contract:{path}",
            "version": version, "repository": "PaddlePaddle/PaddleOCR", "commit": COMMITS[version],
            "sha256": ("a" if version == "v2.9.1" else "b") * 64, "path": path,
            "source_url": f"https://github.com/PaddlePaddle/PaddleOCR/blob/{COMMITS[version]}/{path}"}


SOURCES = [source(version) for version in COMMITS]
CHUNKS = [{**row, "chunk_id": row["source_id"] + ":1", "document_path": row["path"],
           "language": "zh", "line_start": 1, "line_end": 30, "content": "官方接口导出契约", "heading": "接口导出"}
          for row in SOURCES]


def report(files=FILES):
    return {"schema_version": 1, "project_id": "paddleocr", "workspace_id": "paddleocr",
            "source_version": "v2.9.1", "target_version": "v3.0.0", "status": "supported_risk", "runtime_verified": False,
            "files": [{"path": row["path"], "sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                       "line_count": len(row["content"].splitlines())} for row in files],
            "findings": [{"finding_id": "finding-1", "rule_id": "removed_structure_api", "status": "supported_risk",
                          "title": "PPStructure 导出接口变化", "explanation": "目标版本的导出入口发生变化。",
                          "application": {"path": files[0]["path"], "line": 1, "end_line": 1,
                                          "snippet": files[0]["content"].splitlines()[0]},
                          "evidence": [{"source_id": row["source_id"], "chunk_id": row["chunk_id"],
                                        "version": row["version"], "sha256": row["sha256"], "commit": row["commit"],
                                        "url": row["source_url"], "path": row["path"], "line_start": 5,
                                        "line_end": 7, "title": "接口导出"} for row in CHUNKS],
                          "desired_check": "核对文档解析入口与下游输出"}],
            "gaps": [], "verification_steps": ["用真实扫描文档进行 OCR 推理回归，当前未验证"],
            "summary": {"finding_count": 1, "risk_count": 1, "gap_count": 0}}


class Gateway:
    def __init__(self):
        self.payload = report()
        self.workspace_payload = {"workspace_id": "paddleocr", "domain_profile": {"id": "paddleocr"},
                                  "repository": "PaddlePaddle/PaddleOCR", "repositories": ["PaddlePaddle/PaddleOCR"],
                                  "languages": ["zh"], "available_versions": list(COMMITS), "source_registry": SOURCES,
                                  "rag_ready": True, "corpus_fingerprint": {"fingerprint_sha256": "c" * 64}}
        self.chunks = copy.deepcopy(CHUNKS)
        self.calls = []
        self.advice = {"status": "OK", "claim_verification":{"status":"SUPPORTED"}, "review": {"review_status": "REQUIRES_HUMAN_REVIEW", "impact_candidates": [
            {"evidence_chunk_id": CHUNKS[1]["chunk_id"], "reason": "目标版本接口需复核", "suggested_action": "运行文档样本回归"}]}}

    def workspace(self):
        return self.workspace_payload

    def compatibility_review(self, files, **kwargs):
        self.calls.append(("tool", kwargs))
        return self.payload

    def document(self, document_id):
        self.calls.append(("document", document_id))
        return self.chunks

    def search(self, question, **kwargs):
        self.calls.append(("search", kwargs))
        return {"results": [row for row in self.chunks if row["version"] == kwargs["version"]]}

    def review_advice_for_version(self, summary, evidence_chunk_ids, *, version,request_timeout=None):
        self.advice_summary = summary
        self.calls.append(("model", version, evidence_chunk_ids))
        return self.advice


def test_profile_and_controlled_review_bind_app_and_both_official_versions():
    assert load_change_profile(PROFILE_PATH)["id"] == "paddleocr"
    gateway = Gateway()
    result = PaddleOCRReviewAgent(gateway).analyze(FILES)
    assert result["compatibility_report"] == gateway.payload
    assert result["workspace_id"] == "paddleocr"
    assert result["runtime_verified"] is False
    assert result["model_calls"] == 0
    assert result["manual_review_required"] is True
    assert len(result["request_fingerprint"]) == 64
    assert {call[1]["version"] for call in gateway.calls if call[0] == "search"} == set(COMMITS)
    assert "实际 OCR 推理未验证" in result["rendered_report"]


def test_investigation_records_each_checked_item_and_tool_budget():
    result=PaddleOCRReviewAgent(Gateway()).analyze(FILES)
    assert result['investigation']['search_calls']==2
    assert result['investigation']['trace']
    assert result['compatibility_report']['status']=='supported_risk'


def test_long_valid_static_report_uses_bounded_advice_summary_without_losing_report():
    gateway = Gateway()
    finding = gateway.payload["findings"][0]
    gateway.payload["findings"] = [
        {**copy.deepcopy(finding), "finding_id": f"finding-{i}", "explanation": "需核对接口结果消费" * 50}
        for i in range(100)
    ]
    gateway.payload["summary"].update(finding_count=100, risk_count=100)
    result = PaddleOCRReviewAgent(gateway).analyze(FILES, generate_advice=True)
    assert len(gateway.advice_summary) <= 4000
    assert "摘要截断" in gateway.advice_summary
    assert "v2.9.1" in gateway.advice_summary and "v3.0.0" in gateway.advice_summary
    assert "app/scan.py:1" in gateway.advice_summary
    assert "同一 evidence_chunk_id" in gateway.advice_summary
    assert len(result["compatibility_report"]["findings"]) == 100
    assert result["model_status"] == "OK"


@pytest.mark.parametrize("version", ["latest", "v2.9.0", "2.9.1"])
def test_unpublished_or_alias_request_versions_are_rejected_before_tool(version):
    gateway = Gateway()
    with pytest.raises(ValueError):
        PaddleOCRReviewAgent(gateway).analyze(FILES, source_version=version)
    assert gateway.calls == []


@pytest.mark.parametrize("patch", [{"workspace_id": "pphuman"}, {"repository": "other/project"},
                                   {"languages": ["zh", "en"]}, {"rag_ready": False}])
def test_mismatched_workspaces_never_reach_static_tool(patch):
    gateway = Gateway()
    gateway.workspace_payload.update(patch)
    with pytest.raises(ValueError):
        PaddleOCRReviewAgent(gateway).analyze(FILES)
    assert gateway.calls == []


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(runtime_verified=True),
    lambda value: value.update(project_id="other"),
    lambda value: value["files"][0].update(sha256="0" * 64),
    lambda value: value["findings"][0]["application"].update(snippet="invented"),
    lambda value: value["findings"][0]["evidence"][0].update(commit="0" * 40),
    lambda value: value["findings"][0]["evidence"][0].update(line_end=100),
    lambda value: value["findings"][0].update(evidence=[]),
    lambda value: value["findings"][0].update(evidence=value["findings"][0]["evidence"][:1]),
    lambda value: value.update(status="unaffected"),
])
def test_unbound_or_unsupported_tool_claims_are_rejected(mutate):
    gateway = Gateway()
    mutate(gateway.payload)
    with pytest.raises(ValueError):
        PaddleOCRReviewAgent(gateway).analyze(FILES)
    assert not any(call[0] == "model" for call in gateway.calls)


@pytest.mark.parametrize("path", ["../scan.py", "/scan.py", "C:/scan.py", "app/../scan.py", "app\\scan.py"])
def test_path_traversal_is_rejected_without_executing_code(path):
    gateway = Gateway()
    with pytest.raises(ValueError):
        PaddleOCRReviewAgent(gateway).analyze([{**FILES[0], "path": path}])
    assert gateway.calls == []


def test_optional_model_advice_cannot_modify_static_facts_and_uses_only_target_evidence():
    gateway = Gateway()
    gateway.advice["review"]["impact_candidates"].append({
        "evidence_chunk_id": CHUNKS[0]["chunk_id"], "reason": "旧版引用不应通过", "suggested_action": "伪造已验证"})
    result = PaddleOCRReviewAgent(gateway).analyze(FILES, generate_advice=True)
    assert result["model_calls"] is None
    assert result["model_advice_request_count"] == 1
    assert result["compatibility_report"] == gateway.payload
    assert len(result["model_suggestions"]) == 1
    assert result["model_suggestions"][0]["verified"] is False
    assert next(call for call in gateway.calls if call[0] == "model")[1:] == ("v3.0.0", [CHUNKS[1]["chunk_id"]])


def test_model_failure_leaves_static_report_and_human_review_available():
    gateway = Gateway()
    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    gateway.review_advice_for_version = fail
    result = PaddleOCRReviewAgent(gateway).analyze(FILES, generate_advice=True)
    assert result["model_status"] == "GENERATION_PROVIDER_UNAVAILABLE"
    assert result["compatibility_report"] == gateway.payload
    assert result["model_suggestions"] == []

def test_resuming_does_not_repeat_completed_advice_request():
    gateway=Gateway()
    first=PaddleOCRReviewAgent(gateway).analyze(FILES,generate_advice=True)
    resumed=PaddleOCRReviewAgent(gateway).analyze(FILES,generate_advice=True,resume=first['resume_state'])
    assert len([c for c in gateway.calls if c[0]=='model'])==1
    assert resumed['model_advice_request_count']==1
    assert resumed['model_suggestions']==first['model_suggestions']

def test_resuming_does_not_repeat_failed_advice_request():
    gateway=Gateway();calls=[]
    def fail(*args,request_timeout=None,**kwargs):
        calls.append(1)
        raise RuntimeError('provider unavailable')
    gateway.review_advice_for_version=fail
    first=PaddleOCRReviewAgent(gateway).analyze(FILES,generate_advice=True)
    resumed=PaddleOCRReviewAgent(gateway).analyze(FILES,generate_advice=True,resume=first['resume_state'])
    assert len(calls)==1
    assert resumed['model_advice_request_count']==1
    assert resumed['model_status']=='GENERATION_PROVIDER_UNAVAILABLE'


def test_impact_path_cannot_start_in_an_unrelated_application():
    from app.paddleocr_review import _validate_impact_extension
    text='def first():\n    return second()\ndef second():\n    return 1'
    digest=hashlib.sha256(text.encode()).hexdigest()
    def loc(lo,hi):return {'path':'unrelated.py','line_start':lo,'line_end':hi,'sha256':digest,'excerpt':'\n'.join(text.splitlines()[lo-1:hi])}
    nodes=[{'id':'one','application':loc(1,2)},{'id':'two','application':loc(3,4)}]
    edge={'source':'one','target':'two','kind':'result_flow','application':loc(2,2)}
    value={'impact_schema_version':1,'application_graph':{'nodes':nodes,'edges':[edge]},
           'impact_paths':[{'finding_id':'risk','origin_node':'one','target_node':'two','level':'candidate',
                            'runtime_verified':False,'edges':[edge],'flow_proven':True,'application':loc(3,4)}]}
    with pytest.raises(ValueError,match='origin'):
        _validate_impact_extension(value,{'unrelated.py':text,'risk.py':'pass'},
                                 [{'finding_id':'risk','application':{'path':'risk.py','line':1}}])


def test_related_lookup_failure_does_not_hide_static_evidence():
    gateway = Gateway()
    def fail(*args, **kwargs):
        raise RuntimeError("lookup unavailable")
    gateway.search = fail
    result = PaddleOCRReviewAgent(gateway).analyze(FILES)
    assert len(result["lookup_gaps"]) == 2
    assert result["stage_status"]["official_lookup"] == "partial"
    assert len(result["retrieved_results"]) == 2


@pytest.mark.parametrize("filename,status", [
    ("removed_structure_api.py", "supported_risk"),
    ("legacy_result_consumer.py", "supported_risk"),
    ("compatible_basic.py", "unaffected"),
])
def test_original_examples_bind_actual_pinned_corpus_tool_and_agent(filename, status):
    workspace_root = Path(__file__).resolve().parents[2]
    service_root = workspace_root / "versioned-rag-service"
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))
    from src.paddleocr_compatibility import review_compatibility
    corpus = service_root / "public_corpus_paddleocr"
    index = SimpleNamespace(
        manifest=json.loads((corpus / "corpus_manifest.json").read_text(encoding="utf-8")),
        chunks=json.loads((corpus / "chunks.json").read_text(encoding="utf-8")),
    )
    class PinnedGateway(Gateway):
        def workspace(self):
            return {**self.workspace_payload, "source_registry": index.manifest["sources"]}
        def compatibility_review(self, files, request_timeout=None, **versions):
            return review_compatibility(index, files=files, **versions)
        def document(self, source_id):
            return [row for row in index.chunks if row["source_id"] == source_id or row["document_id"] == source_id]
        def search(self, query, **scope):
            return {"results": []}
    sample = workspace_root / "examples" / "paddleocr_document_app" / filename
    result = PaddleOCRReviewAgent(PinnedGateway()).analyze([{"path": f"app/{filename}", "content": sample.read_text(encoding="utf-8")}])
    assert result["status"] == status
    assert result["runtime_verified"] is False
    assert result["retrieved_results"]
    assert result["model_calls"] == 0
