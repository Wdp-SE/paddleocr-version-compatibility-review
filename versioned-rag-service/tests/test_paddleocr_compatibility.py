from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.paddleocr_compatibility import review_compatibility


CORPUS = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"


@pytest.fixture
def index():
    # These are the actual imported, commit-pinned official sources, not synthetic facts.
    return SimpleNamespace(
        manifest=json.loads((CORPUS / "corpus_manifest.json").read_text(encoding="utf-8")),
        chunks=json.loads((CORPUS / "chunks.json").read_text(encoding="utf-8")),
    )


@pytest.fixture
def guard_index():
    # Boundary validation does not require any compatibility facts.
    return SimpleNamespace(manifest={"workspace_id": "paddleocr", "repository": "PaddlePaddle/PaddleOCR"}, chunks=[])


def review(index, content, path="app.py", **versions):
    return review_compatibility(
        index,
        source_version=versions.get("source_version", "v2.9.1"),
        target_version=versions.get("target_version", "v3.0.0"),
        files=[{"path": path, "content": content}],
    )


def assert_bound(report, index, content):
    sources = {source["source_id"]: source for source in index.manifest["sources"]}
    chunks = {chunk["chunk_id"]: chunk for chunk in index.chunks}
    assert report["runtime_verified"] is False
    assert report["model_call_count"] == 0
    assert report["files"][0]["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    for finding in report["findings"]:
        app = finding["application"]
        assert content.splitlines()[app["line"] - 1] in app["snippet"]
        assert finding["desired_check"]
        assert finding["evidence"]
        for evidence in finding["evidence"]:
            source = sources[evidence["source_id"]]
            chunk = chunks[evidence["chunk_id"]]
            assert chunk["source_id"] == source["source_id"]
            assert evidence["sha256"] == source["sha256"] == chunk["source_sha256"]
            assert evidence["commit"] == source["commit"] == chunk["commit"]
            assert evidence["version"] == source["version"] == chunk["version"]
            assert evidence["url"] == source["source_url"]
            assert evidence["path"] == source["document_path"]
            assert evidence["line_start"] == chunk["line_start"]
            assert evidence["line_end"] == chunk["line_end"]


@pytest.mark.parametrize("constructor", [
    "from paddleocr import PPStructure as Layout\nengine = Layout()",
    "import paddleocr as po\nengine = po.PPStructure()",
    "from paddleocr import PPStructure\nLayout = PPStructure\nengine = Layout()",
])
def test_removed_ppstructure_api_is_a_bound_risk_for_import_aliases(index, constructor):
    report = review(index, constructor)
    assert report["status"] == "supported_risk"
    finding = next(item for item in report["findings"] if item["rule_id"] == "removed_ppstructure")
    assert finding["status"] == "supported_risk"
    assert "PPStructureV3" in finding["explanation"]
    assert_bound(report, index, constructor)


def test_old_nested_result_consumer_has_a_bound_output_contract_risk(index):
    content = (
        "from paddleocr import PaddleOCR\n"
        "engine = PaddleOCR(lang='ch')\n"
        "result = engine.ocr('scan.png')\n"
        "for line in result[0]:\n"
        "    text = line[1][0]\n"
        "    confidence = line[1][1]\n"
    )
    report = review(index, content)
    finding = next(item for item in report["findings"] if item["rule_id"] == "legacy_ocr_result")
    assert report["status"] == "supported_risk"
    assert finding["application"]["line"] == 5
    assert {item["version"] for item in finding["evidence"]} == {"v2.9.1", "v3.0.0"}
    assert_bound(report, index, content)


def test_basic_lang_and_ocr_surface_is_statically_supported_without_inference_claim(index):
    content = "from paddleocr import PaddleOCR as OCR\nengine = OCR(lang='ch')\nresult = engine.ocr('scan.png')\n"
    report = review(index, content, source_version="2.9.1", target_version="3.0.0")
    assert report["source_version"] == "v2.9.1"
    assert report["target_version"] == "v3.0.0"
    assert report["status"] == "unaffected"
    assert {item["rule_id"] for item in report["findings"]} == {"basic_ocr_surface"}
    assert report["gaps"] == []
    assert any("推理" in step for step in report["verification_steps"])
    assert_bound(report, index, content)


@pytest.mark.parametrize("content,code", [
    ("from paddleocr import PaddleOCR\nengine=PaddleOCR(**options)\nengine.ocr('x')", "dynamic_arguments"),
    ("from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch')\ngetattr(engine, method)('x')", "dynamic_access"),
    ("from paddleocr import PaddleOCR\nengine = factory('PaddleOCR')\nengine.ocr('x')", "unresolved_call"),
    ("from paddleocr import *\nengine = PaddleOCR()", "wildcard_import"),
    ("from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch', made_up_option=True)\nengine.ocr('x')", "unreviewed_parameter"),
])
def test_dynamic_or_unreviewed_calls_are_gaps_not_false_safe(index, content, code):
    report = review(index, content)
    assert report["status"] == "needs_verification"
    assert code in {gap["code"] for gap in report["gaps"]}
    assert report["runtime_verified"] is False


@pytest.mark.parametrize("content", [
    "from paddleocr import PaddleOCR\nPaddleOCR = custom_factory\nengine = PaddleOCR()",
    "from paddleocr import PaddleOCR\ndef run(PaddleOCR):\n    engine = PaddleOCR()",
    "import paddleocr as po\npo = custom_module\nengine = po.PPStructure()",
    "from paddleocr import PaddleOCR\nif flag:\n    PaddleOCR = custom_factory\nengine = PaddleOCR()",
    "from paddleocr import PPStructure\n(lambda PPStructure: PPStructure())(custom_factory)",
    "from paddleocr import PPStructure\n[PPStructure() for PPStructure in factories]",
    "from paddleocr import PPStructure\n(PPStructure := custom_factory)\nPPStructure()",
    "from paddleocr import PPStructure\nPPStructure += other\nPPStructure()",
    "from paddleocr import PPStructure\ndef run():\n    PPStructure()\n    from other import PPStructure",
])
def test_rebinding_and_unknown_control_flow_cannot_create_supported_claims(index, content):
    report = review(index, content)
    assert report["status"] == "needs_verification"
    assert not report["findings"]
    assert report["gaps"]


def test_a_single_top_level_result_index_is_not_mislabeled_as_old_nested_contract(index):
    report = review(index, "from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch')\nresult=engine.ocr('x')\nfirst=result[0]\n")
    assert not any(item["rule_id"] == "legacy_ocr_result" for item in report["findings"])


@pytest.mark.parametrize("consumer", [
    "for box, (text, score) in result[0]:\n    print(text, score)",
    "for [box, [text, score]] in result[0]:\n    print(text, score)",
    "[text for box, (text, score) in result[0]]",
    "first, second = result\nfor box, (text, score) in first:\n    print(text, score)",
    "first, *rest = result\nfor box, (text, score) in rest[0]:\n    print(text, score)",
    "first, *middle, last = result\nfor box, (text, score) in middle[0]:\n    print(text, score)",
])
def test_legacy_result_destructuring_is_an_evidence_bound_risk(index, consumer):
    content = "from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch')\nresult=engine.ocr('scan.png')\n" + consumer
    report = review(index, content)
    assert report["status"] == "supported_risk"
    assert any(item["rule_id"] == "legacy_ocr_result" for item in report["findings"])
    assert_bound(report, index, content)


@pytest.mark.parametrize("rebind", [
    "h = Holder()", "h += replacement", "(h := Holder())", "del h",
    "h.parent = Holder()", "h.parent += replacement", "import unrelated as h",
])
def test_rebinding_an_attribute_parent_invalidates_its_ocr_instance_alias(index, rebind):
    nested = "h.parent" in rebind
    instance = "h.parent.engine" if nested else "h.engine"
    content = f"from paddleocr import PaddleOCR\n{instance}=PaddleOCR(lang='ch')\n{rebind}\n{instance}.ocr('scan.png')\n"
    report = review(index, content)
    assert report["status"] == "needs_verification"
    assert not report["findings"]
    assert "unresolved_call" in {gap["code"] for gap in report["gaps"]}


@pytest.mark.parametrize("setup,expression,rebind", [
    ("from paddleocr import PaddleOCR", "PaddleOCR", "PaddleOCR = Local"),
    ("from paddleocr import PaddleOCR as OCR", "OCR", "OCR = Local"),
    ("import paddleocr as po", "po.PaddleOCR", "po = Local"),
    ("import paddleocr as po", "po.PaddleOCR", "po.PaddleOCR = Local"),
    ("from paddleocr import PaddleOCR", "PaddleOCR", "PaddleOCR += Local"),
    ("from paddleocr import PaddleOCR", "PaddleOCR", "(PaddleOCR := Local)"),
])
def test_later_module_rebinding_prevents_stale_global_support_inside_functions(index, setup, expression, rebind):
    content = f"{setup}\ndef run():\n    return {expression}(lang='ch').ocr('scan.png')\n{rebind}\nrun()\n"
    report = review(index, content)
    assert report["status"] == "needs_verification"
    assert not report["findings"]
    assert "deferred_global_binding" in {gap["code"] for gap in report["gaps"]}


def test_unrebound_global_alias_in_original_function_remains_narrowly_supported(index):
    report = review(index, "from paddleocr import PaddleOCR\ndef run():\n    return PaddleOCR(lang='ch').ocr('scan.png')\nrun()\n")
    assert report["status"] == "unaffected"
    assert report["gaps"] == []


def test_destructuring_only_the_top_level_result_list_does_not_assert_old_page_shape(index):
    report = review(index, "from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch')\nresult=engine.ocr('scan.png')\nfirst, second = result\n")
    assert report["status"] == "unaffected"
    assert not any(item["rule_id"] == "legacy_ocr_result" for item in report["findings"])


def test_top_level_starred_result_unpacking_alone_does_not_assert_old_page_shape(index):
    report = review(index, "from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch')\nresult=engine.ocr('scan.png')\nfirst, *rest = result\n")
    assert report["status"] == "unaffected"
    assert not any(item["rule_id"] == "legacy_ocr_result" for item in report["findings"])


def test_no_ocr_usage_and_syntax_errors_do_not_prove_safety(index):
    no_usage = review(index, "value = 1\n")
    syntax_error = review(index, "from paddleocr import (\n")
    assert no_usage["status"] == syntax_error["status"] == "needs_verification"
    assert no_usage["gaps"] and syntax_error["gaps"]


def test_missing_or_unbound_evidence_closes_supported_findings(index):
    broken = SimpleNamespace(manifest=copy.deepcopy(index.manifest), chunks=[])
    report = review(broken, "from paddleocr import PPStructure\nengine=PPStructure()")
    assert report["status"] == "needs_verification"
    assert not report["findings"]
    assert "missing_official_evidence" in {gap["code"] for gap in report["gaps"]}
    tampered = SimpleNamespace(manifest=copy.deepcopy(index.manifest), chunks=copy.deepcopy(index.chunks))
    for chunk in tampered.chunks:
        chunk["source_sha256"] = "0" * 64
    report = review(tampered, "from paddleocr import PPStructure\nengine=PPStructure()")
    assert not report["findings"]
    assert report["status"] == "needs_verification"


def test_ch_basic_surface_requires_bound_target_language_selection_evidence(index):
    index.chunks = [chunk for chunk in index.chunks if not (
        chunk["version"] == "v3.0.0"
        and chunk["document_path"] == "paddleocr/_pipelines/ocr.py"
        and 'lang == "ch"' in chunk["content"]
    )]
    report = review(index, "from paddleocr import PaddleOCR\nengine=PaddleOCR(lang='ch')\nengine.ocr('x')")
    assert report["status"] == "needs_verification"
    assert not report["findings"]
    assert "missing_official_evidence" in {gap["code"] for gap in report["gaps"]}


@pytest.mark.parametrize("versions", [
    {"source_version": "v2.9.0"}, {"target_version": "v3.1.0"},
    {"source_version": "latest"}, {"source_version": "v3.0.0", "target_version": "v2.9.1"},
])
def test_unsupported_version_pairs_are_rejected(guard_index, versions):
    with pytest.raises(ValueError, match="version pair"):
        review(guard_index, "value=1", **versions)


def test_foreign_project_index_is_rejected(guard_index):
    guard_index.manifest["project_id"] = "some-other-project"
    with pytest.raises(ValueError, match="PaddleOCR project"):
        review(guard_index, "value=1")


def test_project_id_without_paddleocr_workspace_cannot_bypass_domain_validation(guard_index):
    guard_index.manifest["project_id"] = "paddleocr"
    guard_index.manifest["workspace_id"] = "other"
    with pytest.raises(ValueError, match="PaddleOCR project"):
        review(guard_index, "value=1")


def test_report_output_is_bounded_and_truncation_is_not_a_clean_bill_of_health(index):
    content = "from paddleocr import PPStructure\n" + "engine = PPStructure()\n" * 300
    report = review(index, content)
    assert len(report["findings"]) <= 100
    assert "report_limit" in {gap["code"] for gap in report["gaps"]}


@pytest.mark.parametrize("path", ["../secret.py", "a/../secret.py", "C:\\secret.py", "/secret.py", "a//b.py", "\\\\server\\x.py", "NUL.py"])
def test_application_paths_are_labels_and_reject_traversal_or_device_paths(guard_index, path):
    with pytest.raises(ValueError, match="path"):
        review(guard_index, "value=1", path=path)


def test_count_bytes_lines_and_parser_depth_are_bounded(guard_index):
    with pytest.raises(ValueError, match="file count"):
        review_compatibility(guard_index, source_version="v2.9.1", target_version="v3.0.0", files=[])
    with pytest.raises(ValueError, match="file count"):
        review_compatibility(guard_index, source_version="v2.9.1", target_version="v3.0.0", files=[{"path": f"{i}.py", "content": "x=1"} for i in range(13)])
    with pytest.raises(ValueError, match="bytes"):
        review(guard_index, "#" + "汉" * 20000)
    with pytest.raises(ValueError, match="lines"):
        review(guard_index, "x=1\n" * 5001)
    with pytest.raises(ValueError, match="nesting"):
        review(guard_index, "x=" + "[" * 129 + "0" + "]" * 129)
    with pytest.raises(ValueError, match="AST"):
        review(guard_index, "x=[" + ",".join("0" for _ in range(15001)) + "]")


def test_config_dependency_mismatch_is_explicit_without_yaml_execution(guard_index):
    report = review(guard_index, "paddleocr==3.0.0\n", path="requirements.txt")
    assert report["status"] == "needs_verification"
    assert "dependency_version_mismatch" in {gap["code"] for gap in report["gaps"]}
    yaml_report = review(guard_index, "factory: !!python/object/apply:os.system [whoami]\n", path="pipeline.yaml")
    assert yaml_report["status"] == "needs_verification"
    assert "unreviewed_configuration" in {gap["code"] for gap in yaml_report["gaps"]}


def test_pasted_source_is_never_executed_or_loaded_from_application_path(index, tmp_path):
    marker = tmp_path / "executed"
    content = f"open({str(marker)!r}, 'w').write('wrong')\nfrom paddleocr import PPStructure\nengine=PPStructure()\n"
    report = review(index, content, path="nonexistent/app.py")
    assert report["status"] == "supported_risk"
    assert not marker.exists()
