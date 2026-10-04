"""One-hop, provenance-bound references in indexed PP-Human configuration/docs.

This is a textual reference tool, not a YAML executor or a code impact graph.
Missing files remain gaps; they are never fetched or read from another index.
"""

from __future__ import annotations

import ast
import hashlib
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


MAX_ROOTS = 8
MAX_RELATIONS = 80
MAX_GAPS = 80
MAX_SOURCE_BYTES = 1_000_000
MAX_VERSION_SOURCES = 200
_REPOSITORY = "PaddlePaddle/PaddleDetection"
_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_URL = re.compile(r"https?://[^\s<>`\]\)]+")
_PATH = re.compile(r"(?<![A-Za-z0-9_./:\\-])((?:\./|\.\./)*[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:ya?ml|md))(?![A-Za-z0-9_.-])", re.I)
_BASE = re.compile(r"^_BASE_\s*:\s*(.*)$")


def _relative_path(value: str, owner: str, *, root_relative: bool | None = None) -> str | None:
    value = unquote(value.split("#", 1)[0].split("?", 1)[0])
    if not value or "\\" in value or "\x00" in value or ":" in value or value.startswith("/"):
        return None
    plain = value[2:] if value.startswith("./") else value
    inferred_root = root_relative is None and plain.startswith(("configs/", "deploy/", "docs/"))
    candidate = plain if root_relative is True or inferred_root else posixpath.join(posixpath.dirname(owner), value)
    candidate = posixpath.normpath(candidate)
    if candidate in (".", "..") or candidate.startswith("../"):
        return None
    return candidate


def _base_values(raw: str) -> list[str] | None:
    raw = raw.split("#", 1)[0].strip().rstrip(",")
    if raw.startswith("[") or raw.startswith(("'", '"')):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, str):
            parsed = [parsed]
        return parsed if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed) else None
    if re.fullmatch(r"[A-Za-z0-9_./:-]+\.ya?ml", raw):
        return [raw]
    return None


def _quoted_evidence(lines: list[str], start: int, end: int, value: str | None = None) -> dict:
    original = "\n".join(lines[start:end + 1])
    position = original.find(value) if value else 0
    offset = max(0, position - 80) if position >= 600 or position + len(value or "") > 600 else 0
    quote = original[offset:offset + 600]
    line_start = start + 1 + original[:offset].count("\n")
    return {"line_start": line_start, "line_end": line_start + quote.count("\n"), "text": quote}


def _references(body: str, path: str):
    """Yield literal declarations with source lines, without interpreting YAML."""
    lines = body.splitlines()
    is_yaml = path.lower().endswith((".yml", ".yaml"))
    consumed = set()
    for offset, line in enumerate(lines):
        if not is_yaml or offset in consumed:
            continue
        match = _BASE.match(line)
        if not match:
            continue
        declaration, end = match.group(1).strip(), offset
        if declaration.startswith("["):
            while "]" not in declaration and end + 1 < len(lines) and end - offset < 20:
                end += 1
                declaration += "\n" + lines[end]
            values = _base_values(declaration)
        elif not declaration or declaration.startswith("#"):
            values = []
            while end + 1 < len(lines):
                next_line = lines[end + 1]
                if not next_line.strip() or next_line.lstrip().startswith("#"):
                    end += 1
                    continue
                entry = re.match(r"^\s+-\s+(.+?)\s*$", next_line)
                if not entry:
                    break
                end += 1
                parsed = _base_values(entry.group(1))
                if parsed is None:
                    values = None
                    break
                values.extend(parsed)
            if not values:
                values = None
        else:
            values = _base_values(declaration)
        consumed.update(range(offset, end + 1))
        evidence = _quoted_evidence(lines, offset, end)
        if values is None:
            yield None, "unsupported_base_syntax", evidence, False
        else:
            for value in values:
                yield value, "config_inherits", _quoted_evidence(lines, offset, end, value), False
    seen = set()
    for offset, line in enumerate(lines):
        if offset in consumed or line.lstrip().startswith("#") and is_yaml:
            continue
        kind = "config_references" if is_yaml else "document_references"
        remaining = line
        for match in _LINK.finditer(line):
            value = match.group(1)
            if urlparse(value).path.lower().endswith((".yml", ".yaml", ".md")):
                key = (value, kind)
                if key not in seen:
                    seen.add(key)
                    yield value, kind, _quoted_evidence(lines, offset, offset, value), False
            remaining = remaining.replace(match.group(0), " " * len(match.group(0)))
        # External URL tails must not become apparently local paths.
        remaining = _URL.sub(" ", remaining)
        for match in _PATH.finditer(remaining):
            value = match.group(1)
            key = (value, kind)
            if key not in seen:
                seen.add(key)
                yield value, kind, _quoted_evidence(lines, offset, offset, value), None


def _target(value: str, source: dict, version: str, root_relative: bool | None) -> tuple[str, str | None]:
    if re.match(r"^[A-Za-z]:[/\\]", value):
        return "invalid_path", value
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme != "https" or parsed.hostname != "github.com":
            return "external_repository", value
        parts = unquote(parsed.path).strip("/").split("/")
        if len(parts) < 5 or "/".join(parts[:2]).casefold() != _REPOSITORY.casefold():
            return "external_repository", value
        if parts[2] != "blob" or parts[3] not in (source.get("commit"), version):
            return "different_version", value
        normalized = _relative_path("/".join(parts[4:]), source["document_path"], root_relative=True)
    else:
        normalized = _relative_path(value, source["document_path"], root_relative=root_relative)
    return ("candidate", normalized) if normalized is not None else ("invalid_path", value)


def _source_info(source: dict, document_id: str) -> dict:
    return {"source_id": source["source_id"], "document_id": document_id,
        "path": source["document_path"], "version": source["version"],
        "source_url": source["source_url"]}


def trace_configuration(index, document_ids: list[str], version: str) -> dict:
    """Return same-version outgoing and incoming literal references for 1–8 roots.

    `index` supplies validated corpus `root`, `manifest`, and `chunks`. Source
    hash, project boundary and chunk-to-source binding are checked again here.
    Raises ValueError for unknown/cross-version roots or drifted provenance.
    """
    if not isinstance(document_ids, list) or not 1 <= len(document_ids) <= MAX_ROOTS or any(not isinstance(item, str) or not item for item in document_ids):
        raise ValueError("configuration trace requires 1 to 8 document roots")
    manifest = index.manifest
    if manifest.get("workspace_id") != "pphuman" or manifest.get("repository") != _REPOSITORY:
        raise ValueError("configuration trace requires the PP-Human primary project")
    if manifest.get("dependency_reference_allowlist") or manifest.get("dependency_reference_count", 0):
        raise ValueError("dependency references require a separate reviewed index")
    selected = manifest.get("current_version") if version == "latest" else version
    if selected not in manifest.get("available_versions", []):
        raise ValueError("unknown configuration trace version")
    registered_commit = manifest.get("versions", {}).get(selected, {}).get("commit")
    if not isinstance(registered_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", registered_commit):
        raise ValueError("configuration trace version commit is not registered")
    root_path = Path(index.root).resolve()
    rows, paths, bodies = {}, {}, {}
    sources = [source for source in manifest.get("sources", []) if source.get("version") == selected]
    if len(sources) > MAX_VERSION_SOURCES:
        raise ValueError("configuration trace source budget exceeded")
    chunk_ids = {}
    source_paths = {source.get("source_id"): source.get("document_path") or source.get("path") for source in sources}
    for chunk in index.chunks:
        if chunk.get("version") == selected:
            if chunk.get("repository") != _REPOSITORY or chunk.get("document_path") != source_paths.get(chunk.get("source_id")):
                raise ValueError("configuration trace chunk provenance mismatch")
            chunk_ids.setdefault(chunk.get("source_id"), set()).add(chunk.get("document_id"))
    for source in sources:
        path = source.get("document_path") or source.get("path")
        if (source.get("repository") != _REPOSITORY or source.get("namespace", "project_primary") != "project_primary"
                or source.get("project_id", manifest.get("project_id")) != manifest.get("project_id")
                or not isinstance(path, str) or _relative_path(path, "", root_relative=True) != path
                or not isinstance(source.get("source_id"), str)):
            raise ValueError("configuration trace source provenance mismatch")
        commit = source.get("commit")
        source_url = source.get("source_url", "")
        if commit != registered_commit or source_url != f"https://github.com/{_REPOSITORY}/blob/{commit}/{path}":
            raise ValueError("configuration trace source commit/URL mismatch")
        local_path = source.get("local_path")
        if not isinstance(local_path, str) or "\\" in local_path or "\x00" in local_path:
            raise ValueError("configuration trace source local path is invalid")
        local = (root_path / local_path).resolve()
        if not local.is_relative_to(root_path) or not local.is_file() or local.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("configuration trace source path/budget mismatch")
        raw = local.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.get("sha256"):
            raise ValueError("configuration trace source hash mismatch")
        document_id = f"{selected}:{source.get('language')}:{source.get('document_key')}"
        if chunk_ids.get(source["source_id"]) != {document_id} or document_id in rows or path in paths:
            raise ValueError("configuration trace source/chunk binding mismatch")
        rows[document_id] = dict(source, document_path=path)
        paths[path] = document_id
        bodies[document_id] = raw.decode("utf-8")
    root_ids = list(dict.fromkeys(document_ids))
    if any(document_id not in rows for document_id in root_ids):
        raise ValueError("unknown or cross-version configuration trace root")
    relations, gaps = [], []
    seen_relations = set()
    truncated = False
    for document_id, source in rows.items():
        for value, kind, evidence, root_relative in _references(bodies[document_id], source["document_path"]):
            if value is None:
                if document_id in root_ids:
                    if len(gaps) < MAX_GAPS:
                        gaps.append({**_source_info(source, document_id), "kind": kind, "evidence": evidence})
                    else:
                        truncated = True
                continue
            target_status, target_path = _target(value, source, selected, root_relative)
            target_id = paths.get(target_path) if target_status == "candidate" else None
            if target_status == "candidate":
                target_status = "indexed" if target_id else "not_indexed"
            if document_id not in root_ids and target_id not in root_ids:
                continue
            relation_key = (document_id, kind, target_status, target_path)
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)
            if len(relations) >= MAX_RELATIONS:
                truncated = True
                continue
            relation = {**_source_info(source, document_id), "relation_type": kind,
                "direction": "outgoing" if document_id in root_ids else "incoming",
                "target_path": target_path, "target_document_id": target_id,
                "target_source_id": rows[target_id]["source_id"] if target_id else None,
                "target_status": target_status, "evidence": evidence}
            relations.append(relation)
            if target_status != "indexed":
                if len(gaps) < MAX_GAPS:
                    gaps.append({**relation, "kind": target_status})
                else:
                    truncated = True
    return {"schema_version": 1, "tool": "public_config_trace", "version": selected,
        "roots": [_source_info(rows[item], item) for item in root_ids],
        "relations": relations, "gaps": gaps,
        "status": "partial" if gaps or truncated else "ok" if relations else "no_relations",
        "bounds": {"max_roots": MAX_ROOTS, "max_relations": MAX_RELATIONS, "max_gaps": MAX_GAPS,
            "depth": 1, "scanned_sources": len(sources), "truncated": truncated},
        "boundary_note": "仅表示已入库同版本资料的显式路径关系；不证明实际运行影响，不执行配置，不读取缺失或外部文件。"}
