"""PaddleOCR provenance boundaries and structural chunking regression tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def _module():
    try:
        return importlib.import_module("src.paddleocr_corpus")
    except ModuleNotFoundError as exc:
        pytest.fail(f"PaddleOCR corpus implementation is missing: {exc}")


def test_markdown_chunks_keep_fences_tables_and_exact_source_lines():
    # Character splitting a fence or table would remove the API/result context.
    body = "# 中文 OCR\n\n参数说明。\n\n```python\n" + "value = '" + "x" * 150 + "'\n" + "# 假标题不应成为章节\n```\n\n| 参数 | 类型 |\n| --- | --- |\n| lang | str |\n| device | str |\n\n## 输出\n\n结果说明。"
    chunks = _module().structural_chunks(body, "markdown", max_chars=80)
    lines = body.splitlines()
    for chunk in chunks:
        assert chunk["content"] == "\n".join(lines[chunk["line_start"] - 1:chunk["line_end"]])
    fence = next(c for c in chunks if "```python" in c["content"])
    assert fence["content"].endswith("```")
    assert "# 假标题不应成为章节" in fence["content"]
    table = next(c for c in chunks if "| 参数 | 类型 |" in c["content"])
    assert "| device | str |" in table["content"]
    assert next(c for c in chunks if "结果说明" in c["content"])["heading_path"] == ["中文 OCR", "输出"]


def test_python_and_yaml_chunks_preserve_semantic_blocks():
    python = "class PaddleOCR:\n    def ocr(self, img):\n        result = []\n        for line in img:\n            result.append(line)\n        return result\n\nclass PPStructure:\n    pass\n"
    chunks = _module().structural_chunks(python, "python", max_chars=35)
    method = next(c for c in chunks if "def ocr" in c["content"])
    assert "return result" in method["content"]
    assert "class PPStructure" not in method["content"]
    yaml = "Global:\n  use_gpu: true\n  save_res_path: ./output\nArchitecture:\n  model_type: rec\n  algorithm: SVTR\n"
    chunks = _module().structural_chunks(yaml, "yaml", max_chars=20)
    assert chunks[0]["content"] == "Global:\n  use_gpu: true\n  save_res_path: ./output"
    assert chunks[1]["content"] == "Architecture:\n  model_type: rec\n  algorithm: SVTR"


def test_official_html_parameter_tables_remain_one_citable_block():
    body = "# 参数\n\n<table>\n<tr><th>参数</th><th>说明</th></tr>\n\n<tr><td>lang</td><td>语言</td></tr>\n\n<tr><td>device</td><td>设备</td></tr>\n</table>\n\n后续说明。"
    chunks = _module().structural_chunks(body, "markdown", max_chars=30)
    table = next(c for c in chunks if "<table>" in c["content"])
    assert "device" in table["content"]
    assert table["content"].endswith("</table>")
    assert table["block_type"] == "table"


def test_config_rejects_unpinned_release_and_unselected_repository_path(tmp_path):
    module = _module()
    project, selection = module.load_paddleocr_config()
    broken = json.loads(json.dumps(project))
    broken["versions"][list(broken["versions"])[0]]["commit"] = "a" * 40
    with pytest.raises(ValueError, match="pinned"):
        module.validate_paddleocr_config(broken, selection)
    broken_selection = json.loads(json.dumps(selection))
    broken_selection["sources"][0]["path"] = "../README_en.md"
    with pytest.raises(ValueError, match="allowlist|unsafe|selection"):
        module.validate_paddleocr_config(project, broken_selection)


def test_downloader_rejects_foreign_host_and_incorrect_official_blob(tmp_path):
    try:
        builder = importlib.import_module("scripts.build_paddleocr_corpus")
    except ModuleNotFoundError as exc:
        pytest.fail(f"PaddleOCR pinned downloader is missing: {exc}")
    with pytest.raises(ValueError, match="official|URL"):
        builder.fetch_official_bytes("https://example.com/README.md")
    project, selection = _module().load_paddleocr_config()
    # Only the external read is substituted; blob validation remains real.
    root = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"
    def bad_fetch(url):
        if "/git/ref/tags/" in url:
            version = url.rsplit("/", 1)[-1]
            return json.dumps({"object": {"type": "commit", "sha": project["versions"][version]["commit"]}}).encode()
        if url.endswith("/LICENSE"):
            version = next(v for v, row in project["versions"].items() if row["commit"] in url)
            return (root / "provenance" / version / "LICENSE").read_bytes()
        return b"untrusted or truncated bytes"
    with pytest.raises(ValueError, match="blob|license|tag"):
        builder.build_corpus(tmp_path / "bad-corpus", project, selection, fetcher=bad_fetch)
    assert not (tmp_path / "bad-corpus" / "corpus_manifest.json").exists()


def _real_copy(tmp_path):
    module = _module()
    root = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"
    if not (root / "corpus_manifest.json").is_file():
        pytest.fail("Pinned official PaddleOCR corpus has not been built")
    import shutil
    copy = tmp_path / "corpus"
    shutil.copytree(root, copy)
    manifest = json.loads((copy / "corpus_manifest.json").read_text(encoding="utf-8"))
    chunks = json.loads((copy / "chunks.json").read_text(encoding="utf-8"))
    return module, copy, manifest, chunks


def test_real_corpus_hashes_line_coordinates_and_counts_are_verified(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    result = module.validate_paddleocr_manifest(root, manifest, chunks)
    assert result["source_count"] >= 30
    assert result["chinese_document_count"] >= 20
    assert result["python_contract_count"] >= 4
    assert result["yaml_config_count"] >= 2
    assert result["source_count"] == sum(result[k] for k in ("chinese_document_count", "python_contract_count", "yaml_config_count"))
    assert result["current_version"].removeprefix("v") == "3.0.0"
    assert len(chunks) == manifest["chunk_count"]
    assert all(c["line_start"] >= 1 and c["line_end"] >= c["line_start"] for c in chunks)


@pytest.mark.parametrize("field,value", [
    ("commit", "a" * 40), ("repository", "untrusted/PaddleOCR"),
    ("local_path", "../outside.md"), ("path", "README_en.md"),
    ("source_url", "https://example.com/unpinned.md"),
])
def test_manifest_rejects_provenance_drift_and_path_escape(tmp_path, field, value):
    module, root, manifest, chunks = _real_copy(tmp_path)
    manifest["sources"][0][field] = value
    with pytest.raises(ValueError):
        module.validate_paddleocr_manifest(root, manifest, chunks)


def test_manifest_rejects_tampered_source_and_invented_chunk_content(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    chunks[0]["content"] += "\n编造的兼容结论"
    with pytest.raises(ValueError, match="line|content|chunk"):
        module.validate_paddleocr_manifest(root, manifest, chunks)
    source = manifest["sources"][0]
    body = root / source["local_path"]
    body.write_bytes(body.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="hash|blob"):
        module.validate_paddleocr_manifest(root, manifest)


def test_manifest_rejects_invented_headings(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    original = chunks[0]["heading_path"]
    chunks[0]["heading_path"] = ["编造的已验证兼容声明"]
    with pytest.raises(ValueError, match="structur|heading|chunk"):
        module.validate_paddleocr_manifest(root, manifest, chunks)


def test_manifest_rejects_missing_approved_source(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    removed = manifest["sources"].pop()
    manifest["source_count"] -= 1
    manifest[{"markdown":"chinese_document_count", "python":"python_contract_count", "yaml":"yaml_config_count"}[removed["source_format"]]] -= 1
    with pytest.raises(ValueError, match="selection|registry|approved"):
        module.validate_paddleocr_manifest(root, manifest)


def test_manifest_rejects_symlink_parent_even_when_target_inside_corpus(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    source = manifest["sources"][0]
    target = root / "sources-target"
    (root / "sources").rename(target)
    link = root / "sources"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit creating directory symlinks")
    with pytest.raises(ValueError):
        module.validate_paddleocr_manifest(root, manifest, chunks)


def test_rebuild_refuses_reviewed_figures_before_mutating_other_artifacts(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    policy_path = root / "retrieval_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["review_note"] = "保留审阅记录"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    original = policy_path.read_bytes()
    sidecar = root / "figure_evidence_reviewed.json"
    sidecar.write_text('{"schema_version":1,"chunks":[{"chunk_id":"reviewed"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed figure"):
        module.build_paddleocr_index(root)
    assert policy_path.read_bytes() == original


def test_rebuild_keeps_structural_chunks_and_runtime_locks(tmp_path):
    module, root, manifest, chunks = _real_copy(tmp_path)
    stats = module.build_paddleocr_index(root)
    rebuilt = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
    assert rebuilt == chunks
    assert stats["chunks"] == len(chunks)
    digest = hashlib.sha256((root / "corpus_manifest.json").read_bytes()).hexdigest()
    for name in ("figure_evidence.json", "figure_evidence_reviewed.json", "figure_evidence_reviewed.lock.json"):
        assert json.loads((root / name).read_text(encoding="utf-8"))["corpus_manifest_sha256"] == digest
    from src.public_knowledge import PublicKnowledgeIndex
    from src.public_retrieval_runtime import PublicRetrievalRuntime
    index = PublicKnowledgeIndex(root)
    runtime = PublicRetrievalRuntime(index, config_path=root / "public_retrieval_runtime.json")
    hits = runtime.search("PaddleOCR OCR 返回 rec_texts", version="latest", language="zh")
    assert hits
    assert all(hit["version"].removeprefix("v") == "3.0.0" for hit in hits)
