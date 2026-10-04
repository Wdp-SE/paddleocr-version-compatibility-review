"""Server-controlled retrieval policies layered over the immutable V3 index."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from src.public_knowledge import tokens
from src.retrieval_fusion import fuse_ranked_hits, split_query_facets


POLICIES = {
    "bm25", "bm25_faceted_rrf", "bm25_pphuman_term_expansion_rrf", "bm25_figure_ocr",
    "bm25_faceted_figure_ocr", "hybrid", "task_adaptive_rerank",
}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "public_corpus_paddleocr" / "public_retrieval_runtime.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PublicRetrievalRuntime:
    """Adds bounded facets and approved image OCR without mutating V3 artifacts."""

    def __init__(
        self, base_index, *, config_path: Path | None = None,
        sidecar_path: Path | None = None, inventory_path: Path | None = None,
        sidecar_data: dict | None = None, inventory_data: dict | None = None,
    ):
        self.base_index = base_index
        self.root = base_index.root
        self.manifest = base_index.manifest
        self.chunks = base_index.chunks
        self.policy = base_index.policy
        self.config_path = Path(config_path or self.root / "public_retrieval_runtime.json")
        self.sidecar_path = Path(sidecar_path or self.root / "figure_evidence_reviewed.json")
        self.inventory_path = Path(inventory_path or self.root / "figure_evidence.json")
        try:
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.sidecar = (
                sidecar_data if sidecar_data is not None
                else json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            )
            inventory = (
                inventory_data if inventory_data is not None
                else json.loads(self.inventory_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("public retrieval runtime config or image sidecar is unavailable/invalid") from exc
        self._validate_config(self.config)
        manifest_sha = hashlib.sha256((self.root / "corpus_manifest.json").read_bytes()).hexdigest()
        if self.sidecar.get("schema_version") != 1 or self.sidecar.get("corpus_manifest_sha256") != manifest_sha:
            raise ValueError("public image sidecar does not match the pinned corpus manifest")
        if inventory.get("schema_version") != 1 or inventory.get("corpus_manifest_sha256") != manifest_sha:
            raise ValueError("public figure inventory does not match the pinned corpus manifest")
        self._sources = {
            (row["version"], row["language"], row["document_key"]): row
            for row in self.manifest.get("sources", [])
        }
        self._inventory = {row.get("figure_id"): row for row in inventory.get("figures", [])}
        self._images = self._validated_images()
        self.last_retrieval_call_count = 0

    @staticmethod
    def _validate_config(config: dict) -> None:
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValueError("invalid public retrieval runtime config schema")
        allowed = config.get("allowed_policies")
        if not isinstance(allowed, list) or not allowed or len(allowed) != len(set(allowed)):
            raise ValueError("invalid public retrieval runtime config allowlist")
        if any(policy not in POLICIES for policy in allowed):
            raise ValueError("invalid public retrieval runtime config policy")
        if config.get("default_policy") not in allowed:
            raise ValueError("invalid public retrieval runtime config default policy")
        for key, low, high in (("max_facets", 1, 4), ("rrf_k", 1, 1000), ("image_top_k", 1, 20)):
            value = config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"invalid public retrieval runtime config {key}")

    def _validated_images(self) -> list[dict]:
        rows = []
        for source_row in self.sidecar.get("chunks", []):
            if not isinstance(source_row, dict):
                continue
            figure_id = source_row.get("figure_id")
            figure = self._inventory.get(figure_id)
            source = self._sources.get((
                source_row.get("version"), source_row.get("language"), source_row.get("document_key"),
            ))
            if (
                source_row.get("schema_version") != 1
                or source_row.get("modality") != "image_ocr"
                or source_row.get("review_status") != "approved"
                or not isinstance(source_row.get("content"), str)
                or not source_row["content"].strip()
                or not isinstance(source_row.get("chunk_id"), str)
                or not source_row["chunk_id"].strip()
                or not isinstance(figure, dict)
                or not isinstance(source, dict)
            ):
                continue
            sha = source_row.get("sha256", "")
            if (
                not _SHA256_RE.fullmatch(sha)
                or figure.get("validation", {}).get("status") != "verified"
                or figure.get("validation", {}).get("sha256") != sha
                or figure.get("version") != source_row.get("version")
                or figure.get("commit") != source_row.get("commit")
                or figure.get("raw_url") != source_row.get("raw_url")
                or source.get("source_url") != source_row.get("source_url")
            ):
                continue
            references = figure.get("references", [])
            if not any(
                ref.get("document_key") == source_row.get("document_key")
                and ref.get("language") == source_row.get("language")
                and ref.get("heading") == source_row.get("heading")
                for ref in references
            ):
                continue
            rows.append({
                **source_row,
                "document_id": f"{source_row['version']}:{source_row['language']}:{source_row['document_key']}",
                "locale": source["locale"],
                "source_type": source["source_type"],
                "repository": source.get("repository"),
                "document_path": source.get("document_path"),
                **{
                    field: source[field]
                    for field in (
                        "device_model", "module_sku", "carrier_board", "software_baselines",
                        "source_snapshot", "source_id", "license", "license_status", "attribution",
                    )
                    if source.get(field)
                },
                "retrieval_policy": "bm25_figure_ocr",
            })
        return rows

    @property
    def runtime_policy(self) -> str:
        return self.config["default_policy"]

    @property
    def reviewable_chunks(self) -> list[dict]:
        """Evidence accepted by review APIs, including independently validated OCR rows."""
        return [*self.chunks, *self._images]

    def _version_members(self, version: str) -> set[str] | None:
        """Expose the base index's exact/composite scope resolver to API consumers."""
        return self.base_index._version_members(version)

    def _image_scope(
        self, *, version: str, language: str, device_model: str | None = None,
        module_sku: str | None = None, carrier_board: str | None = None,
        software_baseline: str | None = None,
    ) -> list[dict]:
        try:
            version_members = self.base_index._version_members(version)
        except ValueError as exc:
            raise ValueError("unsupported public corpus scope") from exc
        if language not in ("zh_preferred", "all", "zh", "en"):
            raise ValueError("unsupported public corpus scope")
        eligible = [
            row for row in self._images
            if version_members is None or row["version"] in version_members
        ]
        filters = {
            "device_model": device_model,
            "module_sku": module_sku,
            "carrier_board": carrier_board,
            "software_baselines": software_baseline,
        }
        for field, expected in filters.items():
            if expected is None:
                continue
            eligible = [
                row for row in eligible
                if row.get(field) == "*" or isinstance(row.get(field), list)
                and ("*" in row[field] or expected in row[field])
            ]
        if language in ("zh", "en"):
            return [row for row in eligible if row["language"] == language]
        # One citation per screenshot; zh_preferred and all both avoid duplicate OCR rows.
        preferred = "zh" if any(row["language"] == "zh" for row in eligible) else "en"
        chosen = {}
        for row in eligible:
            key = (row["figure_id"], row["version"])
            if key not in chosen or row["language"] == preferred:
                chosen[key] = row
        return list(chosen.values())

    def _search_images(
        self, query: str, *, top_k: int, version: str, language: str,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
    ) -> list[dict]:
        query_terms = tokens(query)
        if not query_terms:
            return []
        rows = self._image_scope(
            version=version, language=language, device_model=device_model,
            module_sku=module_sku, carrier_board=carrier_board,
            software_baseline=software_baseline,
        )
        if not rows:
            return []
        term_rows = [Counter(tokens(row["heading"] + " " + row["document_key"] + " " + row["content"])) for row in rows]
        doc_freq = Counter()
        for terms in term_rows:
            doc_freq.update(terms.keys())
        lengths = [sum(terms.values()) for terms in term_rows]
        average_length = max(sum(lengths) / len(lengths), 1.0)
        query_freq = Counter(query_terms)
        content_term_rows = [set(tokens(row["content"])) for row in rows]
        scored = []
        total = len(rows)
        for row, freqs, length, content_terms in zip(rows, term_rows, lengths, content_term_rows):
            # Require two distinct matches in reviewed image text itself. Headings and
            # document keys alone are too broad and caused unrelated diagrams to rank.
            content_overlap = set(query_freq) & content_terms
            if len(content_overlap) < 2:
                continue
            score = 0.0
            for term, qf in query_freq.items():
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (total - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += qf * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average_length))
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        return [
            {**row, "rank": rank, "retrieval_score": score, "retrieval_policy": "bm25_figure_ocr"}
            for rank, (score, row) in enumerate(scored[:min(top_k, self.config["image_top_k"])], 1)
        ]

    @staticmethod
    def _add_image_evidence(document_hits: list[dict], image_hits: list[dict], *, top_k: int) -> list[dict]:
        """Add relevant image evidence while preserving source-document coverage."""
        merged = [dict(row) for row in document_hits[:top_k]]
        seen = {row.get("chunk_id") for row in merged}
        for image in image_hits:
            if image.get("chunk_id") in seen:
                continue
            same_source = next((
                index for index in range(len(merged) - 1, -1, -1)
                if merged[index].get("document_id") == image.get("document_id")
            ), None)
            if same_source is None:
                source_counts = Counter(row.get("document_id") for row in merged)
                same_source = next((
                    index for index in range(len(merged) - 1, -1, -1)
                    if source_counts.get(merged[index].get("document_id"), 0) > 1
                ), None)
            if same_source is not None:
                # Replace a duplicate chunk from the same source, or any duplicate
                # source when it is already overrepresented in Top-K. Unique-source
                # coverage is retained even when the Top-K budget is full.
                merged[same_source] = dict(image)
                seen.add(image.get("chunk_id"))
            elif len(merged) < top_k:
                merged.append(dict(image))
                seen.add(image.get("chunk_id"))
            # If adding the image would evict a distinct source, omit it. Evidence
            # coverage takes precedence over image coverage under a strict Top-K cap.
        return merged

    def search(self, query: str, *, top_k: int = 5, version: str = "current",
               language: str = "zh_preferred", policy: str | None = None,
               device_model: str | None = None, module_sku: str | None = None,
               carrier_board: str | None = None, software_baseline: str | None = None,
               source_namespace: str = "project_primary") -> list[dict]:
        selected = policy or self.runtime_policy
        if selected not in self.config["allowed_policies"]:
            raise ValueError("unsupported retrieval runtime policy")
        facet_kwargs = {
            "device_model": device_model,
            "module_sku": module_sku,
            "carrier_board": carrier_board,
            "software_baseline": software_baseline,
            "source_namespace": source_namespace,
        }
        # Delegate validation for query length, top_k, version and language to the pinned index.
        if selected == "bm25":
            self.last_retrieval_call_count = 1
            return self.base_index.search(query, top_k=top_k, version=version, language=language, policy="bm25", **facet_kwargs)
        if selected == "hybrid":
            self.last_retrieval_call_count = 1
            return self.base_index.search(query, top_k=top_k, version=version, language=language, policy="hybrid", **facet_kwargs)
        if selected == "bm25_pphuman_term_expansion_rrf":
            self.last_retrieval_call_count = 1
            return self.base_index.search(
                query, top_k=top_k, version=version, language=language,
                policy=selected, **facet_kwargs,
            )
        if selected == "task_adaptive_rerank":
            raise ValueError("task-adaptive reranking must be orchestrated by the public API")

        use_facets = selected in ("bm25_faceted_rrf", "bm25_faceted_figure_ocr")
        use_images = selected in ("bm25_figure_ocr", "bm25_faceted_figure_ocr")
        facets = split_query_facets(query, max_facets=self.config["max_facets"]) if use_facets else [query]
        # Validate the result size before performing any ranking calls.
        if not 1 <= top_k <= 20:
            raise ValueError("invalid search request")
        document_rankings = []
        image_hits = []
        self.last_retrieval_call_count = 0
        for facet in facets:
            document_rankings.append(self.base_index.search(
                facet, top_k=top_k, version=version, language=language, policy="bm25", **facet_kwargs,
            ))
            self.last_retrieval_call_count += 1
            if use_images:
                image_hits.extend(self._search_images(
                    facet, top_k=top_k, version=version, language=language,
                    device_model=device_model, module_sku=module_sku,
                    carrier_board=carrier_board, software_baseline=software_baseline,
                ))
                self.last_retrieval_call_count += 1
        if len(document_rankings) == 1:
            result = document_rankings[0]
        else:
            result = fuse_ranked_hits(document_rankings, top_k=top_k, rrf_k=self.config["rrf_k"])
        if len(document_rankings) == 1 and not use_images:
            return result
        if use_images:
            image_hits.sort(key=lambda row: (-row.get("retrieval_score", 0.0), row.get("chunk_id", "")))
            result = self._add_image_evidence(result, image_hits, top_k=top_k)
        for row in result:
            row["retrieval_policy"] = selected
        return result
