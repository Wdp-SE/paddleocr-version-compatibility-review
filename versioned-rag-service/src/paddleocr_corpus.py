"""Pinned PaddleOCR source integrity and line-preserving, structural indexing.

Only official, reviewed paths at the two approved commits enter this workspace.
Python is parsed for boundaries but is never imported or executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import stat
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np


PADDLEOCR_WORKSPACE_ID = "paddleocr"
PADDLEOCR_REPOSITORY = "PaddlePaddle/PaddleOCR"
PADDLEOCR_PUBLISHER = "PaddlePaddle"
PADDLEOCR_LICENSE = "Apache-2.0"
PADDLEOCR_VERSIONS = {
    "v2.9.1": {"tag": "v2.9.1", "commit": "07603421c20a96bb94bb87d0c4211032527ae706"},
    "v3.0.0": {"tag": "v3.0.0", "commit": "a8474288ad53c0f439c272b786c5fa6240f0cf27"},
}
SERVICE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = SERVICE_ROOT / "config" / "paddleocr_project.json"
SOURCE_SELECTION = SERVICE_ROOT / "config" / "paddleocr_source_selection.json"
PADDLEOCR_SOURCE_PATHS = {
    "v2.9.1": frozenset({
        "README.md", "doc/doc_ch/quickstart.md", "doc/doc_ch/whl.md",
        "doc/doc_ch/inference_args.md", "doc/doc_ch/inference_ppocr.md",
        "doc/doc_ch/inference.md", "doc/doc_ch/algorithm_inference.md",
        "doc/doc_ch/config.md", "doc/doc_ch/installation.md",
        "doc/doc_ch/FAQ.md", "doc/doc_ch/multi_languages.md",
        "doc/doc_ch/angle_class.md", "doc/doc_ch/visualization.md",
        "doc/doc_ch/table_recognition.md", "ppstructure/README_ch.md",
        "ppstructure/docs/quickstart.md", "ppstructure/docs/inference.md",
        "ppstructure/layout/README_ch.md", "ppstructure/table/README_ch.md",
        "ppstructure/recovery/README_ch.md", "ppstructure/return_word_pos.md",
        "paddleocr.py", "__init__.py", "ppstructure/predict_system.py",
        "configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml",
        "configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml", "configs/table/SLANet_ch.yml",
    }),
    "v3.0.0": frozenset({
        "README.md", "docs/quick_start.md", "docs/update/upgrade_notes.md",
        "docs/version3.x/installation.md", "docs/version3.x/pipeline_usage/OCR.md",
        "docs/version3.x/pipeline_usage/PP-StructureV3.md",
        "docs/version3.x/pipeline_usage/doc_preprocessor.md",
        "docs/version3.x/pipeline_usage/table_recognition_v2.md",
        "docs/version3.x/module_usage/text_detection.md",
        "docs/version3.x/module_usage/text_recognition.md",
        "docs/version3.x/module_usage/layout_detection.md",
        "docs/version3.x/module_usage/table_structure_recognition.md",
        "docs/version3.x/module_usage/table_cells_detection.md",
        "docs/version3.x/module_usage/table_classification.md",
        "docs/version3.x/module_usage/doc_img_orientation_classification.md",
        "docs/version3.x/module_usage/text_line_orientation_classification.md",
        "docs/version3.x/module_usage/text_image_unwarping.md",
        "docs/version3.x/deployment/high_performance_inference.md",
        "docs/version3.x/deployment/obtaining_onnx_models.md",
        "docs/version3.x/deployment/on_device_deployment.md",
        "docs/version3.x/deployment/serving.md", "docs/version3.x/logging.md",
        "docs/version3.x/paddleocr_and_paddlex.md",
        "paddleocr/__init__.py", "paddleocr/_pipelines/__init__.py",
        "paddleocr/_pipelines/ocr.py", "paddleocr/_pipelines/base.py",
        "paddleocr/_pipelines/pp_structurev3.py", "paddleocr/_common_args.py",
        "configs/table/SLANet_plus.yml", "configs/table/SLANeXt_wired.yml",
    }),
}
_HAN = re.compile(r"[\u3400-\u9fff]")
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")
_HTML_TABLE = re.compile(r"<table(?:\s|>)", re.I)
_FORMATS = {".md": "markdown", ".py": "python", ".yml": "yaml", ".yaml": "yaml"}
_SOURCE_TYPES = {
    "markdown": "official_documentation", "python": "official_api_contract",
    "yaml": "official_project_config",
}


def json_bytes(payload: object, *, compact: bool = False) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, **(
        {"separators": (",", ":")} if compact else {"indent": 2}
    )) + "\n").encode("utf-8")


def write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload, compact=compact))


def git_blob_sha(raw: bytes) -> str:
    """Verify raw bytes against the blob identity returned by the pinned Git tree."""
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def source_id(version: str, path: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.casefold()).strip("-")
    return f"{version.replace('.', '-')}-{slug[:62]}-{hashlib.sha1(path.encode()).hexdigest()[:8]}"


def _safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value or any(c in value for c in ("\\", ":", "\0")):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and value == path.as_posix()


def _is_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def safe_corpus_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    if not _safe_relative(relative):
        raise ValueError("PaddleOCR source path is unsafe")
    root = Path(root).absolute()
    if _is_link(root):
        raise ValueError("PaddleOCR corpus root must not be a symlink or junction")
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if _is_link(candidate):
            raise ValueError("PaddleOCR corpus does not allow symlinks or junctions")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("PaddleOCR source is outside the approved corpus")
    if must_exist and not candidate.is_file():
        raise ValueError("PaddleOCR source body is missing")
    return candidate


def validate_paddleocr_config(project: dict, selection: dict) -> dict:
    if (not isinstance(project, dict) or project.get("schema_version") != 1
        or project.get("workspace_id") != PADDLEOCR_WORKSPACE_ID
        or project.get("repository") != PADDLEOCR_REPOSITORY
        or project.get("publisher") != PADDLEOCR_PUBLISHER
        or project.get("license") != PADDLEOCR_LICENSE):
        raise ValueError("PaddleOCR project identity or license mismatch")
    if project.get("versions") != PADDLEOCR_VERSIONS or list(project["versions"]) != list(PADDLEOCR_VERSIONS):
        raise ValueError("PaddleOCR releases must match the exact pinned commits")
    if project.get("current_version") != "v3.0.0":
        raise ValueError("PaddleOCR latest must be the latest pinned version v3.0.0")
    if not isinstance(selection, dict) or selection.get("schema_version") != 1 or selection.get("workspace_id") != PADDLEOCR_WORKSPACE_ID:
        raise ValueError("PaddleOCR source selection identity mismatch")
    rows = selection.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("PaddleOCR source selection is empty")
    registry = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("PaddleOCR source selection record is invalid")
        version, path = row.get("version"), row.get("path")
        if not isinstance(version, str) or not _safe_relative(path) or path not in PADDLEOCR_SOURCE_PATHS.get(version, ()):
            raise ValueError("PaddleOCR source is outside the reviewed allowlist")
        key = (version, path)
        if key in registry or row.get("commit") != PADDLEOCR_VERSIONS[version]["commit"]:
            raise ValueError("PaddleOCR source selection duplicate or pinned commit mismatch")
        if (not isinstance(row.get("git_blob_sha"), str) or not _SHA40.fullmatch(row["git_blob_sha"])
            or row.get("source_format") != _FORMATS.get(PurePosixPath(path).suffix)
            or not isinstance(row.get("title"), str) or not _HAN.search(row["title"])
            or not isinstance(row.get("document_family"), str) or not row["document_family"]):
            raise ValueError("PaddleOCR source selection format, Chinese title or blob identity is invalid")
        registry[key] = row
    if {key[0] for key in registry} != set(PADDLEOCR_VERSIONS):
        raise ValueError("PaddleOCR source selection must cover both approved releases")
    return registry


def load_paddleocr_config(project_path: Path | None = None, selection_path: Path | None = None) -> tuple[dict, dict]:
    try:
        project = json.loads(Path(project_path or PROJECT_CONFIG).read_text(encoding="utf-8-sig"))
        selection = json.loads(Path(selection_path or SOURCE_SELECTION).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("PaddleOCR approved project config is unavailable or invalid") from exc
    validate_paddleocr_config(project, selection)
    return project, selection


def _trimmed_block(lines: list[str], start: int, end: int, headings: list[str], kind: str) -> dict | None:
    while start < end and not lines[start].strip():
        start += 1
    while start < end and not lines[end - 1].strip():
        end -= 1
    if start == end:
        return None
    return {"heading": headings[-1] if headings else "", "heading_path": list(headings),
            "line_start": start + 1, "line_end": end, "block_type": kind,
            "content": "\n".join(lines[start:end])}


def structural_chunks(text: str, source_format: str, max_chars: int = 1800) -> list[dict]:
    """Soft-size limit: fences, tables, Python definitions and YAML mappings stay intact."""
    if source_format not in {"markdown", "python", "yaml"} or max_chars < 1:
        raise ValueError("unsupported structural chunk format or size")
    lines = text.splitlines()
    if not lines:
        return []
    blocks: list[dict] = []
    if source_format == "markdown":
        headings: list[str] = []
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            start = i
            match = _HEAD.match(lines[i])
            fence = _FENCE.match(lines[i])
            kind = "paragraph"
            if match:
                headings = headings[:len(match.group(1)) - 1] + [match.group(2).strip()]
                i += 1
                kind = "heading"
            elif fence:
                marker = fence.group(1)
                i += 1
                while i < len(lines):
                    closing = _FENCE.match(lines[i])
                    i += 1
                    if closing and closing.group(1)[0] == marker[0] and len(closing.group(1)) >= len(marker):
                        break
                kind = "code_fence"
            elif i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEPARATOR.match(lines[i + 1]):
                i += 2
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    i += 1
                kind = "table"
            elif _HTML_TABLE.search(lines[i]):
                while i < len(lines):
                    closing = "</table>" in lines[i].casefold()
                    i += 1
                    if closing:
                        break
                kind = "table"
            else:
                i += 1
                while i < len(lines) and lines[i].strip() and not _HEAD.match(lines[i]) and not _FENCE.match(lines[i]) and not _HTML_TABLE.search(lines[i]):
                    if i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEPARATOR.match(lines[i + 1]):
                        break
                    i += 1
            block = _trimmed_block(lines, start, i, headings, kind)
            if block:
                blocks.append(block)
    else:
        boundaries: dict[int, list[str]] = {0: []}
        if source_format == "python":
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                raise ValueError("official Python source cannot be parsed for structural boundaries") from exc
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
                    boundaries[start] = [node.name]
                    if isinstance(node, ast.ClassDef):
                        for member in node.body:
                            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                start = min([member.lineno, *[d.lineno for d in member.decorator_list]]) - 1
                                boundaries[start] = [node.name, member.name]
        else:
            for i, line in enumerate(lines):
                match = re.match(r"^([A-Za-z_][\w.-]*):", line)
                if match:
                    boundaries[i] = [match.group(1)]
        positions = sorted(boundaries)
        for start, end in zip(positions, positions[1:] + [len(lines)]):
            block = _trimmed_block(lines, start, end, boundaries[start], "python_definition" if source_format == "python" else "yaml_mapping")
            if block:
                blocks.append(block)
    chunks = []
    current = None
    for block in blocks:
        atomic = block["block_type"] in {"code_fence", "table", "python_definition", "yaml_mapping"}
        if current is not None and (atomic or current["heading_path"] != block["heading_path"] or len(current["content"]) + len(block["content"]) > max_chars):
            chunks.append(current)
            current = None
        if atomic:
            chunks.append(block)
        elif current is None:
            current = dict(block)
        else:
            current["line_end"] = block["line_end"]
            current["content"] = "\n".join(lines[current["line_start"] - 1:current["line_end"]])
            current["block_type"] = "section"
    if current:
        chunks.append(current)
    return chunks


def _counts(sources: list[dict]) -> dict:
    counts = Counter(row["source_format"] for row in sources)
    return {"chinese_document_count": counts["markdown"], "python_contract_count": counts["python"], "yaml_config_count": counts["yaml"]}


def validate_paddleocr_manifest(root: Path, manifest: dict, chunks: list[dict] | None = None) -> dict:
    """Reject identity drift, unregistered paths, symlinks and invented chunk text."""
    project, selection = load_paddleocr_config()
    registry = validate_paddleocr_config(project, selection)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
        or manifest.get("workspace_id") != PADDLEOCR_WORKSPACE_ID
        or manifest.get("repository") != PADDLEOCR_REPOSITORY
        or manifest.get("publisher") != PADDLEOCR_PUBLISHER
        or manifest.get("license") != PADDLEOCR_LICENSE
        or manifest.get("languages") != ["zh"] or manifest.get("project_id")):
        raise ValueError("PaddleOCR corpus identity, license or Chinese-only workspace mismatch")
    if (manifest.get("versions") != PADDLEOCR_VERSIONS
        or list(manifest["versions"]) != list(PADDLEOCR_VERSIONS)
        or manifest.get("available_versions") != list(PADDLEOCR_VERSIONS)
        or manifest.get("current_version") != project["current_version"]
        or manifest.get("baseline_version") != "v2.9.1"
        or (manifest.get("version_scopes") or {}).get("latest") != {"versions": ["v3.0.0"]}):
        raise ValueError("PaddleOCR release registry must match the approved pinned commits")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("PaddleOCR source registry is empty")
    by_id, seen, bodies, structures = {}, set(), {}, {}
    for row in sources:
        if not isinstance(row, dict) or not isinstance(row.get("version"), str) or not isinstance(row.get("path"), str):
            raise ValueError("PaddleOCR source record is invalid")
        version, path = row["version"], row["path"]
        approved = registry.get((version, path))
        if approved is None or (version, path) in seen:
            raise ValueError("PaddleOCR source is outside the reviewed allowlist or duplicated")
        commit = PADDLEOCR_VERSIONS[version]["commit"]
        fmt = approved["source_format"]
        expected = {
            "source_id": source_id(version, path), "source_snapshot": version,
            "document_path": path, "local_path": f"sources/{version}/{path}",
            "document_key": str(PurePosixPath(path).with_suffix("")),
            "repository": PADDLEOCR_REPOSITORY, "publisher": PADDLEOCR_PUBLISHER,
            "commit": commit, "git_blob_sha": approved["git_blob_sha"],
            "source_format": fmt, "source_type": _SOURCE_TYPES[fmt],
            "source_content_language": "zh-CN" if fmt == "markdown" else fmt,
            "language": "zh", "locale": "zh-CN", "license": PADDLEOCR_LICENSE,
            "license_status": "redistributable",
            "source_url": f"https://github.com/{PADDLEOCR_REPOSITORY}/blob/{commit}/{quote(path, safe='/')}",
            "license_url": f"https://github.com/{PADDLEOCR_REPOSITORY}/blob/{commit}/LICENSE",
            "document_family": approved["document_family"], "title": approved["title"],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError("PaddleOCR source provenance does not match the approved pinned registry")
        if not isinstance(row.get("attribution"), str) or not all(term in row["attribution"] for term in ("PaddleOCR", version, commit, "Apache-2.0")):
            raise ValueError("PaddleOCR source attribution is missing")
        file_path = safe_corpus_path(root, row["local_path"])
        raw = file_path.read_bytes()
        digest = row.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"PaddleOCR source hash mismatch: {row['local_path']}")
        if git_blob_sha(raw) != approved["git_blob_sha"] or row.get("size_bytes") != len(raw):
            raise ValueError("PaddleOCR source Git blob or byte count mismatch")
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("PaddleOCR source is not UTF-8 text") from exc
        if fmt == "markdown" and (len(_HAN.findall(content)) < 20 or not _HAN.search(approved["title"])):
            raise ValueError("PaddleOCR documentation must contain actual Chinese prose and a Chinese title")
        seen.add((version, path))
        by_id[row["source_id"]] = row
        bodies[row["source_id"]] = content.splitlines()
        if chunks is not None:
            doc_id = f"{version}:zh:{row['document_key']}"
            structures.update({f"{doc_id}:{i}": piece for i, piece in enumerate(structural_chunks(content, fmt), 1)})
    if seen != set(registry):
        raise ValueError("PaddleOCR source registry does not cover the approved source selection")
    if manifest.get("source_count") != len(sources) or any(manifest.get(k) != v for k, v in _counts(sources).items()):
        raise ValueError("PaddleOCR document/code/config counts do not match the source registry")
    if {row["version"] for row in sources} != set(PADDLEOCR_VERSIONS):
        raise ValueError("PaddleOCR corpus must have sources for both registered releases")
    if chunks is not None:
        if not isinstance(chunks, list) or manifest.get("chunk_count") != len(chunks) or not chunks:
            raise ValueError("PaddleOCR chunk count does not match the manifest")
        chunk_ids = set()
        for chunk in chunks:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("source_id"), str):
                raise ValueError("PaddleOCR chunk record is invalid")
            source = by_id.get(chunk["source_id"])
            fields = ("version", "language", "locale", "repository", "publisher", "commit", "document_path", "local_path", "source_format", "source_type", "license", "license_url", "license_status", "source_url", "document_key")
            if source is None or any(chunk.get(f) != source.get(f) for f in fields) or chunk.get("source_sha256") != source["sha256"]:
                raise ValueError("PaddleOCR chunk has unregistered source provenance")
            start, end = chunk.get("line_start"), chunk.get("line_end")
            if (type(start) is not int or type(end) is not int or start < 1 or end < start
                or end > len(bodies[source["source_id"]])
                or chunk.get("content") != "\n".join(bodies[source["source_id"]][start - 1:end])
                or not chunk.get("content", "").strip()):
                raise ValueError("PaddleOCR chunk line coordinates or content do not match source bytes")
            if not isinstance(chunk.get("chunk_id"), str) or chunk["chunk_id"] in chunk_ids:
                raise ValueError("PaddleOCR chunk identity is invalid or duplicated")
            expected_document = f"{source['version']}:zh:{source['document_key']}"
            if chunk.get("document_id") != expected_document or not chunk["chunk_id"].startswith(expected_document + ":"):
                raise ValueError("PaddleOCR chunk document identity does not match source")
            structure = structures.get(chunk["chunk_id"])
            if structure is None or any(chunk.get(k) != v for k, v in structure.items()):
                raise ValueError("PaddleOCR chunk structural boundaries or headings do not match the official source")
            chunk_ids.add(chunk["chunk_id"])
        if chunk_ids != set(structures):
            raise ValueError("PaddleOCR chunks must cover every registered source block")
    return {"workspace_id": PADDLEOCR_WORKSPACE_ID, "repository": PADDLEOCR_REPOSITORY,
            "current_version": "v3.0.0", "available_versions": list(PADDLEOCR_VERSIONS),
            "source_count": len(sources), **_counts(sources)}


def _dense_vector(text: str, dimension: int = 512) -> np.ndarray:
    # Same deterministic character-hashing baseline as PublicKnowledgeIndex.
    normalized = "".join(c.casefold() for c in unicodedata.normalize("NFKC", text) if not c.isspace())
    vector = np.zeros(dimension, dtype=np.float32)
    for width in (1, 2, 3):
        for i in range(max(0, len(normalized) - width + 1)):
            digest = hashlib.sha256(normalized[i:i + width].encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % dimension] += 1
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector


def build_paddleocr_index(root: Path) -> dict:
    """Rebuild safely without generic paragraph splitting or model API calls."""
    root = Path(root)
    manifest_path = safe_corpus_path(root, "corpus_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_paddleocr_manifest(root, manifest)
    for name, field in (("figure_evidence.json", "figures"), ("figure_evidence_reviewed.json", "chunks")):
        path = safe_corpus_path(root, name, must_exist=False)
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get(field):
            raise ValueError("PaddleOCR rebuild refuses to replace reviewed figure evidence")
    chunks = []
    for source in manifest["sources"]:
        text = safe_corpus_path(root, source["local_path"]).read_bytes().decode("utf-8-sig")
        doc_id = f"{source['version']}:zh:{source['document_key']}"
        for number, piece in enumerate(structural_chunks(text, source["source_format"]), 1):
            chunks.append({
                **{k: v for k, v in source.items() if k not in {"sha256", "title", "size_bytes"}},
                "chunk_id": f"{doc_id}:{number}", "document_id": doc_id,
                "source_sha256": source["sha256"], "document_title": source["title"], **piece,
            })
    manifest["chunk_count"] = len(chunks)
    validate_paddleocr_manifest(root, manifest, chunks)
    write_json(manifest_path, manifest)
    write_json(safe_corpus_path(root, "chunks.json", must_exist=False), chunks, compact=True)
    vectors = np.stack([_dense_vector(c["heading"] + " " + c["document_key"] + " " + c["content"]) for c in chunks])
    np.save(safe_corpus_path(root, "dense_vectors.npy", must_exist=False), vectors)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(safe_corpus_path(root, "retrieval_policy.json", must_exist=False), {
        "schema_version": 1, "default_policy": "bm25", "selection_status": "new_corpus_pending_rebenchmark",
        "benchmark_query_count": 0, "reranker_enabled": False,
        "benchmark_corpus_sha256": manifest_hash,
        "index_artifacts_sha256": {name: hashlib.sha256(safe_corpus_path(root, name).read_bytes()).hexdigest() for name in ("chunks.json", "dense_vectors.npy")},
    })
    write_json(safe_corpus_path(root, "public_retrieval_runtime.json", must_exist=False), {
        "schema_version": 1, "default_policy": "bm25", "allowed_policies": ["bm25", "bm25_faceted_rrf"],
        "max_facets": 4, "rrf_k": 60, "image_top_k": 5, "corpus_manifest_sha256": manifest_hash,
    })
    for name, field in (("figure_evidence.json", "figures"), ("figure_evidence_reviewed.json", "chunks")):
        path = safe_corpus_path(root, name, must_exist=False)
        write_json(path, {"schema_version": 1, "corpus_manifest_sha256": manifest_hash, field: []})
    write_json(safe_corpus_path(root, "figure_evidence_reviewed.lock.json", must_exist=False), {
        "schema_version": 1, "corpus_manifest_sha256": manifest_hash,
        "sidecar_sha256": hashlib.sha256(safe_corpus_path(root, "figure_evidence_reviewed.json").read_bytes()).hexdigest(),
    })
    write_json(safe_corpus_path(root, "document_relations.json", must_exist=False), {
        "schema_version": 1, "corpus_manifest_sha256": manifest_hash, "relations": [],
    })
    tree_paths = {v: safe_corpus_path(root, f"provenance/tree-{v[1:]}.json", must_exist=False)
                  for v in manifest['versions']}
    if all(p.is_file() for p in tree_paths.values()):
        from src.paddleocr_coverage import audit_coverage
        coverage = audit_coverage(manifest, {v: json.loads(p.read_text(encoding='utf-8'))
                                           for v, p in tree_paths.items()}, root=root)
        write_json(safe_corpus_path(root, 'coverage_manifest.json', must_exist=False), coverage)
    from scripts.build_paddleocr_views import build as build_views_sidecar
    build_views_sidecar(root)
    return {"files": len(manifest["sources"]), "chunks": len(chunks), "dimension": int(vectors.shape[1])}
