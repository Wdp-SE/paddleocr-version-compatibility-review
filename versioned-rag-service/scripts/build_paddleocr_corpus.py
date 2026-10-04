"""Download the official, pinned PaddleOCR Chinese corpus and build its index.

Run from versioned-rag-service: python -m scripts.build_paddleocr_corpus
Reindex without network: python -m scripts.build_paddleocr_corpus --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.paddleocr_corpus import (
    PADDLEOCR_LICENSE, PADDLEOCR_PUBLISHER, PADDLEOCR_REPOSITORY,
    PADDLEOCR_WORKSPACE_ID, SERVICE_ROOT, _SOURCE_TYPES, _counts,
    build_paddleocr_index, git_blob_sha, load_paddleocr_config,
    safe_corpus_path, source_id, validate_paddleocr_config,
    validate_paddleocr_manifest, write_json,
)


DEFAULT_OUTPUT = SERVICE_ROOT / "public_corpus_paddleocr"
MAX_SOURCE_BYTES = 4 * 1024 * 1024


def _approved_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment or parsed.port:
        return False
    if parsed.hostname == "raw.githubusercontent.com":
        return parsed.path.startswith(f"/{PADDLEOCR_REPOSITORY}/") and not parsed.query
    if parsed.hostname == "api.github.com":
        return parsed.path.startswith(f"/repos/{PADDLEOCR_REPOSITORY}/git/")
    return False


class _OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        if not _approved_url(new_url):
            raise ValueError("PaddleOCR download redirected outside official URLs")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def fetch_official_bytes(url: str) -> bytes:
    """Bounded HTTPS read with no credential headers or arbitrary host support."""
    if not _approved_url(url):
        raise ValueError("PaddleOCR download URL must belong to the official repository")
    request = Request(url, headers={"User-Agent": "paddleocr-pinned-corpus/1.0", "Accept": "application/vnd.github+json"})
    opener = build_opener(_OfficialRedirect())
    for attempt in range(3):
        try:
            with opener.open(request, timeout=30) as response:
                if not _approved_url(response.geturl()):
                    raise ValueError("PaddleOCR download response has an unapproved official URL")
                raw = response.read(MAX_SOURCE_BYTES + 1)
                if len(raw) > MAX_SOURCE_BYTES:
                    raise ValueError("PaddleOCR source exceeds the approved byte limit")
                return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise ValueError("PaddleOCR download did not return bytes")


def _raw_url(commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{PADDLEOCR_REPOSITORY}/{commit}/{quote(path, safe='/')}"


def _verify_tag(version: str, snapshot: dict, fetcher) -> dict:
    url = f"https://api.github.com/repos/{PADDLEOCR_REPOSITORY}/git/ref/tags/{snapshot['tag']}"
    try:
        reference = json.loads(fetcher(url))
        obj = reference["object"]
        if obj["type"] == "tag":
            annotated = json.loads(fetcher(f"https://api.github.com/repos/{PADDLEOCR_REPOSITORY}/git/tags/{obj['sha']}"))
            obj = annotated["object"]
        if obj.get("type") != "commit" or obj.get("sha") != snapshot["commit"]:
            raise ValueError("PaddleOCR upstream tag does not match its approved pinned commit")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("PaddleOCR official tag response is invalid") from exc
    return {"version": version, "tag": snapshot["tag"], "commit": obj["sha"], "verification_url": url, "reference": reference}


def build_corpus(output_root: Path, project_config: dict | None = None,
                 source_selection: dict | None = None, *, fetcher=fetch_official_bytes) -> dict:
    """Verify official tags/license/blob identities before registering downloaded sources."""
    if project_config is None or source_selection is None:
        loaded_project, loaded_selection = load_paddleocr_config()
        project_config = loaded_project if project_config is None else project_config
        source_selection = loaded_selection if source_selection is None else source_selection
    registry = validate_paddleocr_config(project_config, source_selection)
    output_root = Path(output_root).absolute()
    safe_corpus_path(output_root, "corpus_manifest.json", must_exist=False)
    versions = project_config["versions"]
    tag_evidence = [_verify_tag(version, snapshot, fetcher) for version, snapshot in versions.items()]
    license_bytes = {}
    for version, snapshot in versions.items():
        raw = fetcher(_raw_url(snapshot["commit"], "LICENSE"))
        if b"Apache License" not in raw or b"Version 2.0" not in raw:
            raise ValueError("PaddleOCR official release license is not Apache-2.0")
        license_bytes[version] = raw
    downloaded = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetcher, _raw_url(row["commit"], row["path"])): key for key, row in registry.items()}
        for future in as_completed(futures):
            key = futures[future]
            raw = future.result()
            approved = registry[key]
            if not raw or len(raw) > MAX_SOURCE_BYTES or git_blob_sha(raw) != approved["git_blob_sha"]:
                raise ValueError(f"PaddleOCR official source blob mismatch: {key[0]}:{key[1]}")
            content = raw.decode("utf-8-sig")
            if approved["source_format"] == "markdown":
                import re
                if len(re.findall(r"[\u3400-\u9fff]", content)) < 20:
                    raise ValueError(f"PaddleOCR selected documentation is not Chinese: {key}")
            downloaded[key] = raw
    sources = []
    for key, approved in registry.items():
        version, path = key
        raw = downloaded[key]
        commit = versions[version]["commit"]
        local_path = f"sources/{version}/{path}"
        destination = safe_corpus_path(output_root, local_path, must_exist=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        fmt = approved["source_format"]
        sources.append({
            "source_id": source_id(version, path), "version": version, "source_snapshot": version,
            "document_key": str(PurePosixPath(path).with_suffix("")), "document_title": approved["title"],
            "document_path": path, "path": path, "local_path": local_path,
            "source_type": _SOURCE_TYPES[fmt], "source_format": fmt,
            "source_content_language": "zh-CN" if fmt == "markdown" else fmt,
            "source_url": f"https://github.com/{PADDLEOCR_REPOSITORY}/blob/{commit}/{quote(path, safe='/')}",
            "raw_url": _raw_url(commit, path), "repository": PADDLEOCR_REPOSITORY,
            "publisher": PADDLEOCR_PUBLISHER, "commit": commit,
            "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_sha": approved["git_blob_sha"],
            "language": "zh", "locale": "zh-CN", "license": PADDLEOCR_LICENSE,
            "license_status": "redistributable",
            "license_url": f"https://github.com/{PADDLEOCR_REPOSITORY}/blob/{commit}/LICENSE",
            "attribution": f"PaddleOCR · {version} · {path} · Apache-2.0 · commit {commit}",
            "document_family": approved["document_family"], "title": approved["title"], "size_bytes": len(raw),
            "scope_note": "官方中文文档" if fmt == "markdown" else "代码/配置辅助证据；不计为中文教程",
        })
    counts = _counts(sources)
    manifest = {
        "schema_version": 1, "workspace_id": PADDLEOCR_WORKSPACE_ID,
        "workspace": project_config["workspace"], "domain_profile": project_config["domain_profile"],
        "repository": PADDLEOCR_REPOSITORY, "publisher": PADDLEOCR_PUBLISHER,
        "license": PADDLEOCR_LICENSE, "license_url": project_config["license_url"],
        "baseline_version": "v2.9.1", "current_version": "v3.0.0", "available_versions": list(versions),
        "version_scopes": {"latest": {"versions": ["v3.0.0"]}}, "versions": versions,
        "languages": ["zh"], "source_count": len(sources), **counts,
        "unique_document_count": len({s["document_key"] for s in sources}), "chunk_count": 0,
        "active": True, "source_status": "ready", "public_body_indexing_enabled": True,
        "retrieval_evaluation_status": "new_corpus_pending_rebenchmark",
        "data_origin": project_config["data_origin"], "upstream_writes_enabled": False,
        "sources": sources,
    }
    validate_paddleocr_manifest(output_root, manifest)
    write_json(safe_corpus_path(output_root, "corpus_manifest.json", must_exist=False), manifest)
    write_json(safe_corpus_path(output_root, "source_import_manifest.json", must_exist=False), {
        "schema_version": 1, "workspace_id": PADDLEOCR_WORKSPACE_ID,
        "repository": PADDLEOCR_REPOSITORY, "publisher": PADDLEOCR_PUBLISHER, "license": PADDLEOCR_LICENSE,
        "latest_version": "v3.0.0", "version_count": 2, "included_count": len(sources), **counts,
        "sources": sources, "excluded": [], "excluded_categories": source_selection.get("excluded_categories", []),
    })
    write_json(safe_corpus_path(output_root, "provenance/release_tags.json", must_exist=False), {"schema_version": 1, "tags": tag_evidence})
    for version, raw in license_bytes.items():
        target = safe_corpus_path(output_root, f"provenance/{version}/LICENSE", must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    result = build_paddleocr_index(output_root)
    audit = ["# PaddleOCR 固定版本语料来源审计", "", project_config["data_origin"], "",
             f"收录 {len(sources)} 个版本化文件：{counts['chinese_document_count']} 篇中文 Markdown、{counts['python_contract_count']} 个 Python 契约、{counts['yaml_config_count']} 个 YAML 配置。代码与 YAML 单独计数。", "",
             "Markdown 保留表格、代码块；Python 使用 AST 定义边界；YAML 保留顶层映射。块长度为软上限，完整大表格或函数可以超过它。每个块内容是原文行切片。", "",
             "两个 tag 的提交由 GitHub Git ref API 验证。原始字节 SHA-256 以及固定 Git tree 的 blob SHA-1 双重校验；tag 元数据、完整树和许可正文保存在 provenance/。", "",
             "Apache-2.0；PaddlePaddle 为原作者。文档中的图片、模型和数据链接仅保留文字，不下载媒体、权重、数据。v3.0.0 的历史 docs/version2.x 不收录，不能借发布提交误判旧 API 的适用版本。", "",
             "当前为新语料，尚未建立冻结评测性能数字。索引中的 dense_vectors 是字符哈希基线；默认策略为 BM25。图片证据是与 manifest 绑定的空审核侧车，不代表运行过 OCR。", "",
             "| 版本 | 类型 | 官方固定提交文件 | 原始字节 SHA-256 |", "| --- | --- | --- | --- |"]
    for row in sources:
        audit.append(f"| {row['version']} | {row['source_format']} | [{row['title']}]({row['source_url']}) | `{row['sha256']}` |")
    safe_corpus_path(output_root, "SOURCE_AUDIT.md", must_exist=False).write_text("\n".join(audit) + "\n", encoding="utf-8", newline="\n")
    return {**result, "workspace_id": PADDLEOCR_WORKSPACE_ID, "source_count": len(sources), **counts,
            "versions": list(versions), "current_version": "v3.0.0",
            "manifest_sha256": hashlib.sha256((output_root / "corpus_manifest.json").read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Rebuild only from already verified local pinned sources")
    args = parser.parse_args()
    result = build_paddleocr_index(args.output) if args.offline else build_corpus(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
