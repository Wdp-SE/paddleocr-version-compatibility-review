"""Inventory official Markdown figures at pinned commits and audit selected images.

This is an offline corpus audit. OCR text is an unreviewed candidate and is not
automatically added to the retrieval index. Markdown alt text is provenance,
never evidence that the words occur inside an image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr"
MAX_IMAGE_BYTES = 2_000_000
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MARKDOWN_IMAGE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<url>[^\s)]+))"
    r"(?:\s+[^)]*)?\)"
)
_HTML_IMAGE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*['\"](?P<url>[^'\"]+)['\"][^>]*>", re.I)
_HTML_ALT = re.compile(r"\balt\s*=\s*['\"](?P<alt>[^'\"]*)['\"]", re.I)
_HEADING = re.compile(r"^\s{0,3}(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")

# The active Seeed snapshot currently has no image references in indexed source pages.
# Keep selection empty until the corpus inventory contains auditable image assets.
SELECTED_FIGURES: tuple[str, ...] = ()


class ImageTooLarge(ValueError):
    """The remote file exceeded the fixed audit size cap."""


def resolve_asset_path(document_path: str, target: str) -> str | None:
    """Resolve a local Markdown target to a repository-relative asset path."""
    if not isinstance(target, str) or not target or "\\" in target:
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
        return None
    decoded = unquote(parts.path)
    if decoded.startswith("/") or "\\" in decoded or "\x00" in decoded:
        return None
    source_path = PurePosixPath(document_path)
    if (
        source_path.is_absolute()
        or any(part in {"", ".", ".."} for part in source_path.parts)
        or not document_path.lower().endswith((".md", ".markdown"))
    ):
        return None
    result = posixpath.normpath(posixpath.join(posixpath.dirname(document_path), decoded))
    if result in {"", ".", ".."} or result.startswith("../"):
        return None
    return result


def _references_in_markdown(text: str):
    headings: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group("marks"))
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, heading.group("title").strip()))
        section = " / ".join(title for _, title in headings)
        for match in _MARKDOWN_IMAGE.finditer(line):
            yield number, match.group("angle") or match.group("url"), match.group("alt"), section
        for match in _HTML_IMAGE.finditer(line):
            alt = _HTML_ALT.search(match.group(0))
            yield number, match.group("url"), alt.group("alt") if alt else "", section


def scan_inventory(root: Path = ROOT) -> dict:
    manifest_bytes = (root / "corpus_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    figures: dict[tuple[str, str], dict] = {}
    relative_count = external_count = unresolved_count = 0
    for source in manifest["sources"]:
        if source.get("source_type") != "official_documentation":
            continue
        repository = source.get("repository") or manifest.get("repository")
        if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            continue
        commit = source.get("commit", "")
        if not _COMMIT.fullmatch(commit):
            raise ValueError("official source has no fixed commit")
        version = source["version"]
        if manifest.get("commits", {}).get(version, commit) != commit:
            raise ValueError("source commit differs from the pinned version")
        local_path = source["local_path"]
        source_file = (root / local_path).resolve()
        if not source_file.is_relative_to(root.resolve()):
            raise ValueError("source path leaves corpus root")
        content = source_file.read_bytes()
        if source.get("sha256") and hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"source hash differs from manifest: {local_path}")
        text = content.decode("utf-8")
        for line, target, alt, heading in _references_in_markdown(text):
            asset_path = resolve_asset_path(source["document_path"], target)
            if asset_path is None:
                if urlsplit(target).scheme or urlsplit(target).netloc:
                    external_count += 1
                else:
                    unresolved_count += 1
                continue
            relative_count += 1
            key = (repository, commit, asset_path)
            if key not in figures:
                figure_id = hashlib.sha256(f"{repository}:{commit}:{asset_path}".encode()).hexdigest()[:16]
                figures[key] = {
                    "schema_version": 1,
                    "figure_id": figure_id,
                    "repository": repository,
                    "version": version,
                    "commit": commit,
                    "asset_path": asset_path,
                    "raw_url": f"https://raw.githubusercontent.com/{repository}/{commit}/{quote(asset_path, safe='/-._')}",
                    "references": [],
                    "validation": {"status": "unverified"},
                    "ocr": {"status": "not_run"},
                }
            figures[key]["references"].append({
                "document_key": source["document_key"],
                "language": source.get("language"),
                "local_path": local_path,
                "line": line,
                "alt_text": alt,
                "heading": heading,
            })
    ordered = sorted(figures.values(), key=lambda row: (row["version"], row["asset_path"]))
    return {
        "schema_version": 1,
        "corpus_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_count": manifest.get("source_count", len(manifest["sources"])),
        "current_version": manifest["current_version"],
        "relative_reference_count": relative_count,
        "external_reference_count": external_count,
        "unresolved_reference_count": unresolved_count,
        "unique_figure_count": len(ordered),
        "search_policy": "OCR text is an unreviewed candidate; alt text is not image content.",
        "figures": ordered,
    }


def _manifest_is_append_only_extension(previous_bytes: bytes, current_bytes: bytes) -> bool:
    """Allow review reuse only when every old pinned source is byte-identical in the new manifest."""
    try:
        previous_manifest = json.loads(previous_bytes)
        current_manifest = json.loads(current_bytes)
    except (json.JSONDecodeError, TypeError):
        return False
    # An unchanged pinned source list may be extended with another release line,
    # locale, or composite current scope without invalidating reviewed old figures.
    # Source rows themselves are compared byte-for-byte below; repository/workspace
    # identity must still match.
    stable_fields = ("repository", "workspace")
    if any(previous_manifest.get(key) != current_manifest.get(key) for key in stable_fields):
        return False
    previous_sources = previous_manifest.get("sources")
    current_sources = current_manifest.get("sources")
    if not isinstance(previous_sources, list) or not isinstance(current_sources, list):
        return False

    def source_map(rows):
        result = {}
        for row in rows:
            if not isinstance(row, dict):
                return None
            identity = (row.get("version"), row.get("document_key"), row.get("language"))
            if not all(identity) or identity in result:
                return None
            result[identity] = row
        return result

    old = source_map(previous_sources)
    new = source_map(current_sources)
    return bool(
        old is not None and new is not None
        and all(new.get(identity) == row for identity, row in old.items())
    )


def preserve_verified_figure_evidence(
    current: dict,
    previous: dict | None,
    *,
    previous_manifest_bytes: bytes | None = None,
    current_manifest_bytes: bytes | None = None,
) -> dict:
    """Retain approvals for unchanged figures across identical or append-only corpora."""
    if not isinstance(previous, dict) or previous.get("schema_version") != 1:
        return current
    same_manifest = previous.get("corpus_manifest_sha256") == current.get("corpus_manifest_sha256")
    append_only = (
        previous_manifest_bytes is not None
        and current_manifest_bytes is not None
        and previous.get("corpus_manifest_sha256") == hashlib.sha256(previous_manifest_bytes).hexdigest()
        and current.get("corpus_manifest_sha256") == hashlib.sha256(current_manifest_bytes).hexdigest()
        and _manifest_is_append_only_extension(previous_manifest_bytes, current_manifest_bytes)
    )
    if not same_manifest and not append_only:
        return current
    prior_rows = {
        row.get("figure_id"): row for row in previous.get("figures", [])
        if isinstance(row, dict) and row.get("figure_id")
    }
    for row in current.get("figures", []):
        prior = prior_rows.get(row.get("figure_id"))
        if not prior or any(prior.get(key) != row.get(key) for key in (
            "repository", "version", "commit", "asset_path", "raw_url", "references",
        )):
            continue
        validation = prior.get("validation")
        sha = validation.get("sha256") if isinstance(validation, dict) else None
        if (
            not isinstance(validation, dict) or validation.get("status") != "verified"
            or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
        ):
            continue
        row["validation"] = dict(validation)
        row["ocr"] = dict(prior.get("ocr") or {"status": "not_run"})
        if isinstance(prior.get("review"), dict):
            row["review"] = dict(prior["review"])
    current["verified_count"] = sum(
        row.get("validation", {}).get("status") == "verified" for row in current.get("figures", [])
    )
    current["ocr_text_count"] = sum(
        row.get("ocr", {}).get("status") == "text_extracted" for row in current.get("figures", [])
    )
    return current


def _fetch_raw_image(url: str, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[bytes, str] | None:
    request = Request(url, headers={"User-Agent": "enterprise-rag-figure-audit/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ImageTooLarge("declared image size exceeds audit cap")
            payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise ImageTooLarge("downloaded image exceeds audit cap")
            return payload, response.headers.get_content_type()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _image_type(payload: bytes) -> str | None:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    stripped = payload.lstrip()
    if stripped[:100].startswith(b"<svg") or stripped.startswith(b"<?xml") and b"<svg" in stripped[:512]:
        return "image/svg+xml"
    return None


def _extract_svg_text(payload: bytes) -> str:
    """Extract explicit SVG text nodes locally; never execute embedded markup."""
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("SVG exceeds the audit size limit")
    upper = payload[: min(len(payload), 8192)].upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("SVG DTD and entity declarations are rejected")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("invalid SVG XML") from exc
    if root.tag.rsplit("}", 1)[-1].casefold() != "svg":
        raise ValueError("document root is not SVG")
    pieces = []

    def add_text(value: str) -> None:
        value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
        value = re.sub(r"<[^>]*>", " ", value)
        text = re.sub(r"\s+", " ", html.unescape(value)).strip()
        text = "".join(character for character in text if character not in "\ufe19\u200b\ufeff")
        if text and text not in pieces:
            pieces.append(text)

    def collect_text(xml_root) -> None:
        for element in xml_root.iter():
            local_name = element.tag.rsplit("}", 1)[-1].casefold()
            if local_name in {"text", "title", "desc"}:
                add_text("".join(element.itertext()))
            if local_name in {"mxcell", "object", "userobject"}:
                for key in ("value", "label"):
                    if element.attrib.get(key):
                        add_text(element.attrib[key])

    collect_text(root)
    embedded = root.attrib.get("content", "")
    if embedded.lstrip().startswith("<"):
        if "<!DOCTYPE" in embedded.upper() or "<!ENTITY" in embedded.upper():
            raise ValueError("embedded Draw.io DTD and entity declarations are rejected")
        try:
            mxfile = ET.fromstring(embedded)
        except ET.ParseError:
            mxfile = None
        if mxfile is not None:
            collect_text(mxfile)
            for diagram in mxfile.iter():
                if diagram.tag.rsplit("}", 1)[-1].casefold() != "diagram" or not diagram.text:
                    continue
                encoded = diagram.text.strip()
                try:
                    compressed = base64.b64decode(encoded, validate=True)
                    inflater = zlib.decompressobj(-15)
                    decoded_bytes = inflater.decompress(compressed, 2_000_001)
                    if len(decoded_bytes) > 2_000_000 or not inflater.eof:
                        continue
                    decoded_xml = unquote(decoded_bytes.decode("utf-8"))
                    if "<!DOCTYPE" in decoded_xml.upper() or "<!ENTITY" in decoded_xml.upper():
                        continue
                    graph = ET.fromstring(decoded_xml)
                except (ValueError, UnicodeDecodeError, zlib.error, ET.ParseError):
                    continue
                collect_text(graph)
    return " ".join(pieces)[:20_000]


def _decode_image(payload: bytes, media_type: str) -> tuple[str, int | None, int | None]:
    """Fully decode raster content when Pillow is installed locally."""
    if media_type == "image/svg+xml":
        _extract_svg_text(payload)
        return "verified", None, None
    try:
        from PIL import Image
    except ImportError:
        return "not_checked", None, None
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if width * height > 25_000_000:
                return "too_many_pixels", width, height
            image.load()
        return "verified", width, height
    except Exception:
        return "invalid", None, None


def _tesseract_path() -> str | None:
    path = shutil.which("tesseract")
    if path:
        return path
    if os.name == "nt":
        candidate = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        if candidate.is_file():
            return str(candidate)
    return None


def run_tesseract_ocr(payload: bytes) -> tuple[str, float | None]:
    executable = _tesseract_path()
    if not executable:
        raise FileNotFoundError("tesseract is unavailable")
    result = subprocess.run(
        [executable, "stdin", "stdout", "-l", "chi_sim+eng", "tsv"],
        input=payload, capture_output=True, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError("tesseract returned a nonzero status")
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    words, confidences = [], []
    for line in lines[1:]:
        cells = line.split("\t", 11)
        if len(cells) != 12 or not cells[11].strip():
            continue
        words.append(cells[11].strip())
        try:
            confidence = float(cells[10])
            if confidence >= 0:
                confidences.append(confidence)
        except ValueError:
            pass
    return " ".join(words), round(sum(confidences) / len(confidences), 2) if confidences else None


def verify_selected_images(
    inventory: dict,
    *,
    selectors: tuple[str, ...] | None = SELECTED_FIGURES,
    fetcher: Callable[[str, int], tuple[bytes, str] | None] = _fetch_raw_image,
    ocr_runner: Callable[[bytes], tuple[str, float | None]] | None = None,
    max_images: int = 20,
    review_dir: Path | None = None,
) -> dict:
    """Fetch a bounded sample; no image binaries or alt-derived OCR are retained."""
    if selectors is None:
        relevant = re.compile(r"(?:device|vision|jetson|flashing|camera|startup)", re.I)
        candidates = [
            row for row in inventory["figures"]
            if row["version"] == inventory["current_version"]
            and PurePosixPath(row["asset_path"]).suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
            and any(relevant.search(ref.get("document_key", "")) for ref in row.get("references", []))
        ]
        candidates.sort(key=lambda row: (
            PurePosixPath(row["asset_path"]).suffix.casefold() == ".svg",
            row["asset_path"],
        ))
        selectors = tuple(row["asset_path"] for row in candidates)
    count = 0
    selector_order = {path: number for number, path in enumerate(selectors or ())}
    ordered_figures = sorted(
        inventory["figures"],
        key=lambda row: selector_order.get(row.get("asset_path"), len(selector_order)),
    )
    for figure in ordered_figures:
        if figure["version"] != inventory["current_version"]:
            continue
        if not any(figure["asset_path"] == selector for selector in selectors):
            continue
        if count >= max_images:
            break
        count += 1
        try:
            fetched = fetcher(figure["raw_url"], MAX_IMAGE_BYTES)
            if fetched is None:
                figure["validation"] = {"status": "not_found"}
                continue
            payload, media_type = fetched
            sniffed = _image_type(payload)
            if not sniffed or sniffed != media_type:
                figure["validation"] = {"status": "invalid_image_content"}
                continue
            decode_status, width, height = _decode_image(payload, sniffed)
            if decode_status in {"invalid", "too_many_pixels"}:
                figure["validation"] = {
                    "status": "invalid_image_content", "decode_status": decode_status,
                }
                continue
            figure["validation"] = {
                "status": "verified" if decode_status == "verified" else "fetched_unchecked",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "media_type": sniffed,
                "decode_status": decode_status,
                "width": width,
                "height": height,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            if review_dir is not None and decode_status == "verified":
                extension = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                    "image/svg+xml": ".svg",
                }.get(sniffed)
                if extension:
                    preview_name = f"{figure['validation']['sha256']}{extension}"
                    review_dir.mkdir(parents=True, exist_ok=True)
                    (review_dir / preview_name).write_bytes(payload)
                    figure["validation"]["review_preview"] = preview_name
            if decode_status != "verified":
                figure["ocr"] = {"status": "not_run"}
                continue
            if ocr_runner is None and sniffed != "image/svg+xml":
                figure["ocr"] = {"status": "tool_unavailable"}
                continue
            try:
                if sniffed == "image/svg+xml":
                    text, confidence = _extract_svg_text(payload), None
                    engine = "svg_text"
                else:
                    text, confidence = ocr_runner(payload)
                    engine = "tesseract"
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                figure["ocr"] = {"status": "ocr_error", "engine": "tesseract"}
                continue
            text = text.strip()
            if text:
                figure["ocr"] = {
                    "status": "text_extracted",
                    "engine": engine,
                    "quality": "unreviewed",
                    "index_review_status": "pending",
                    "mean_confidence": confidence,
                    "text": text,
                }
            else:
                figure["ocr"] = {"status": "no_text_detected", "engine": engine}
        except ImageTooLarge:
            figure["validation"] = {"status": "too_large"}
        except Exception as exc:  # Preserve partial audit results without leaking URLs/proxies.
            figure["validation"] = {"status": "fetch_error", "error_type": type(exc).__name__}
    inventory["verified_count"] = sum(
        row["validation"]["status"] == "verified" for row in inventory["figures"]
    )
    inventory["ocr_text_count"] = sum(
        row["ocr"]["status"] == "text_extracted" for row in inventory["figures"]
    )
    return inventory


def build_reviewed_figure_chunks(rows: list[dict], manifest: dict) -> list[dict]:
    """Build index-ready image evidence only from hash-bound human approvals."""
    commits = manifest.get("commits", {})
    source_repositories = {
        source.get("repository") for source in manifest.get("sources", [])
        if isinstance(source.get("repository"), str)
    }
    repository = manifest.get("repository") or (
        next(iter(source_repositories)) if len(source_repositories) == 1 else None
    )
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        return []
    sources = {
        (source.get("version"), source.get("commit"), source.get("document_key"), source.get("language")): source
        for source in manifest.get("sources", [])
        if source.get("source_type") == "official_documentation"
        and source.get("repository", repository) == repository
    }
    chunks: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        validation = row.get("validation") or {}
        ocr = row.get("ocr") or {}
        review = row.get("review") or {}
        version = row.get("version")
        commit = row.get("commit")
        asset_path = row.get("asset_path")
        asset_repo_path = PurePosixPath(asset_path) if isinstance(asset_path, str) else None
        sha256 = validation.get("sha256")
        if (
            row.get("schema_version") != 1
            or validation.get("status") != "verified"
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or review.get("status") != "approved"
            or review.get("sha256") != sha256
            or ocr.get("status") != "text_extracted"
            or ocr.get("index_review_status") != "approved"
            or row.get("repository") != repository
            or asset_repo_path is None
            or asset_repo_path.is_absolute()
            or any(part in {"", ".", ".."} for part in asset_repo_path.parts)
            or not isinstance(row.get("figure_id"), str)
            or not row.get("figure_id")
            or not isinstance(commit, str)
            or not _COMMIT.fullmatch(commit)
            or (version in commits and commits[version] != commit)
        ):
            continue
        expected_raw = f"https://raw.githubusercontent.com/{repository}/{commit}/{quote(asset_path, safe='/-._')}"
        if row.get("raw_url") != expected_raw:
            continue
        reviewed_by_language = review.get("reviewed_text_by_language") or {}
        for reference in row.get("references", []):
            language = reference.get("language")
            document_key = reference.get("document_key")
            text = reviewed_by_language.get(language, review.get("reviewed_text"))
            if not isinstance(text, str) or not text.strip():
                continue
            source = sources.get((version, commit, document_key, language))
            if not source:
                continue
            source_url = source.get("source_url")
            parsed_source_url = urlsplit(source_url) if isinstance(source_url, str) else None
            if (
                parsed_source_url is None or parsed_source_url.scheme != "https"
                or not parsed_source_url.hostname or not parsed_source_url.path
            ):
                continue
            chunk_id = hashlib.sha256(
                (
                    f"{version}:{language}:{document_key}:{row.get('figure_id')}:{sha256}:"
                    f"{repository}:{reference.get('local_path')}:{reference.get('heading', '')}"
                ).encode()
            ).hexdigest()[:24]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append({
                "schema_version": 1,
                "chunk_id": chunk_id,
                "repository": repository,
                "figure_id": row.get("figure_id"),
                "version": version,
                "commit": commit,
                "language": language,
                "document_key": document_key,
                "heading": reference.get("heading", ""),
                "modality": "image_ocr",
                "review_status": "approved",
                "reviewed_at": review.get("reviewed_at"),
                "review_note": review.get("note", ""),
                "ocr_engine": ocr.get("engine"),
                "ocr_mean_confidence": ocr.get("mean_confidence"),
                "sha256": sha256,
                "content": text.strip(),
                "source_url": source_url,
                "raw_url": expected_raw,
            })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch-selected", action="store_true")
    parser.add_argument("--asset-path", action="append", help="fetch an exact repository-relative figure path; repeatable")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--review-dir", type=Path, help="save verified sample images here for manual review")
    args = parser.parse_args()
    if args.max_images < 0 or args.max_images > 20:
        parser.error("--max-images must be between 0 and 20")
    inventory = scan_inventory(args.root)
    if args.fetch_selected:
        runner = run_tesseract_ocr if _tesseract_path() else None
        verify_selected_images(
            inventory, selectors=tuple(args.asset_path) if args.asset_path else None,
            ocr_runner=runner, max_images=args.max_images,
            review_dir=args.review_dir,
        )
    output = args.output or args.root / "figure_evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_count": inventory["source_count"],
        "relative_references": inventory["relative_reference_count"],
        "unique_figures": inventory["unique_figure_count"],
        "verified": inventory.get("verified_count", 0),
        "ocr_text": inventory.get("ocr_text_count", 0),
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
