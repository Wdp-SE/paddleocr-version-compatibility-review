"""Thin client for the official public-source knowledge workspace."""

from __future__ import annotations

from services.rag_client import RAGClient


class PublicKnowledgeClient(RAGClient):
    def plan_investigation(self,*,unresolved:list[dict],request_timeout:float)->list[dict]:
        items=[{'check_id':r['check_id'],'query':r.get('query','PaddleOCR 结果契约'),'versions':r['versions']} for r in unresolved[:4]]
        result=self._request('POST','/public/investigation-plan',retry_limit=0,request_timeout=request_timeout,
                             json={'items':items,'remaining_seconds':min(240,request_timeout)})
        if result.get('status')!='OK':raise ValueError(result.get('status','PLANNER_INVALID_RESPONSE'))
        return result['queries']
    def workspace(self) -> dict:
        return self._request("GET", "/public/workspace")

    def documents(self) -> list[dict]:
        return self._request("GET", "/public/documents")["documents"]

    def document(self, document_id: str, *, request_timeout: float | None = None) -> list[dict]:
        options={'request_timeout':request_timeout,'retry_limit':0} if request_timeout is not None else {}
        return self._request("POST", "/public/document", json={"document_id": document_id}, **options)["chunks"]

    def compatibility_review(
        self, files: list[dict], *, source_version: str = "v2.9.1",
        target_version: str = "v3.0.0", request_timeout:float|None=None,
    ) -> dict:
        """Submit bounded text to the read-only static compatibility tool."""
        return self._request(
            "POST", "/public/compatibility-review", retry_limit=0,
            request_timeout=request_timeout if request_timeout is not None else max(60.0, self.timeout),
            json={"source_version": source_version, "target_version": target_version, "files": files},
        )

    def config_trace(self, document_ids: list[str], *, version: str) -> dict:
        return self._request("POST", "/public/config-trace", retry_limit=0, json={
            "document_ids": document_ids, "version": version,
        })

    def search(
        self, question: str, *, version: str, language: str = "zh", top_k: int = 5,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
        retrieval_policy: str = "bm25", evidence_strategy: str = 'auto', candidate_budget: int = 80,
        request_timeout: float | None = None,
    ) -> dict:
        payload = {
            "query": question, "version": version, "language": language,
            "top_k": top_k,
        }
        if retrieval_policy != "bm25":
            payload["retrieval_policy"] = retrieval_policy
        if retrieval_policy=='paddleocr_evidence':
            payload.update(evidence_strategy=evidence_strategy,candidate_budget=candidate_budget)
        payload.update({
            key: value for key, value in {
                "device_model": device_model, "module_sku": module_sku,
                "carrier_board": carrier_board, "software_baseline": software_baseline,
            }.items() if value is not None
        })
        return self._request("POST", "/public/search", json=payload,
                             request_timeout=request_timeout if request_timeout is not None else 300.0 if retrieval_policy in ('paddleocr_quality','paddleocr_evidence') else self.timeout,
                             **({'retry_limit':0} if request_timeout is not None else {}))

    def query_official(
        self, question: str, *, version: str, language: str = "zh", top_k: int = 5,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
        retrieval_policy: str = "bm25", evidence_strategy: str = 'auto', candidate_budget: int = 80,
    ) -> dict:
        payload = {
            "query": question, "version": version, "language": language,
            "top_k": top_k,
        }
        if retrieval_policy != "bm25":
            payload["retrieval_policy"] = retrieval_policy
        if retrieval_policy=='paddleocr_evidence':
            payload.update(evidence_strategy=evidence_strategy,candidate_budget=candidate_budget)
        payload.update({
            key: value for key, value in {
                "device_model": device_model, "module_sku": module_sku,
                "carrier_board": carrier_board, "software_baseline": software_baseline,
            }.items() if value is not None
        })
        return self._request(
            "POST", "/public/query", retry_limit=0,
            request_timeout=300.0 if retrieval_policy in ('paddleocr_quality','paddleocr_evidence') else 120.0,
            json=payload,
        )

    def search_batch(
        self, summary: str, *, checks: list[dict], version: str,
        language: str = "zh", top_k: int = 5,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None,
        retrieval_policy: str = "bm25",
    ) -> dict:
        payload = {
            "summary": summary, "checks": checks, "version": version,
            "language": language, "top_k": top_k,
        }
        if retrieval_policy != "bm25":
            payload["retrieval_policy"] = retrieval_policy
        payload.update({
            key: value for key, value in {
                "device_model": device_model, "module_sku": module_sku,
                "carrier_board": carrier_board, "software_baseline": software_baseline,
            }.items() if value is not None
        })
        return self._request(
            "POST", "/public/search-batch", retry_limit=0,
            request_timeout=120.0, json=payload,
        )

    def review_advice(
        self, change_summary: str, evidence_chunk_ids: list[str], *, version: str = "current",
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None, request_timeout:float|None=None,
    ) -> dict:
        payload = {
            "change_summary": change_summary,
            "evidence_chunk_ids": evidence_chunk_ids,
        }
        if version != "current":
            payload["version"] = version
        payload.update({
            key: value for key, value in {
                "device_model": device_model, "module_sku": module_sku,
                "carrier_board": carrier_board, "software_baseline": software_baseline,
            }.items() if value is not None
        })
        return self._request(
            "POST", "/public/review-advice", retry_limit=0, request_timeout=request_timeout if request_timeout is not None else max(60.0, self.timeout),
            json=payload,
        )

    def review_advice_for_version(
        self, change_summary: str, evidence_chunk_ids: list[str], *, version: str,
        device_model: str | None = None, module_sku: str | None = None,
        carrier_board: str | None = None, software_baseline: str | None = None, request_timeout:float|None=None,
    ) -> dict:
        return self.review_advice(
            change_summary, evidence_chunk_ids, version=version,
            device_model=device_model, module_sku=module_sku,
            carrier_board=carrier_board, software_baseline=software_baseline,
            **({'request_timeout':request_timeout} if request_timeout is not None else {}),
        )

    def engineering_diff(self, old_items: list[dict], new_items: list[dict]) -> dict:
        return self._request(
            "POST", "/engineering/items/diff",
            json={"old_items": old_items, "new_items": new_items},
        )

    def engineering_impacts(self, payload: dict) -> dict:
        return self._request("POST", "/engineering/impacts/discover", json=payload)
