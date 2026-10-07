# Internal RAG and Upgrade Review Implementation Plan

> **For agentic workers:** Use executing-plans for Native execution, with TDD and a final independent review.

**Goal:** Make the existing internal document-processing POC consistent in version scope, evidence sufficiency and human risk disposition.
**Architecture:** Keep the shared FastAPI evidence service and bounded review workflow. Add small pure modules for file roles, check coverage and report-bound human disposition; integrate them into the existing Streamlit entry point. Original source identity and static findings remain authoritative.
**Tech Stack:** Python, AST, FastAPI, Streamlit, SQLite, pytest.
**Spec:** docs/2026-10-07-internal-rag-agent-iteration-proposal.md and docs/business_scope.md, approved by the user's instruction to implement the preceding discussion.

## Global Constraints

- Internal users only as business positioning; anonymous public demo remains available.
- Only PaddleOCR v2.9.1 → v3.0.0; OCR → normalization → JSON is the primary application.
- Do not execute submitted code or promote model advice to confirmed facts.
- Preserve existing unrelated uncommitted changes and opened evaluation datasets.
- Actual source additions invalidate prior release fingerprints; show no stale scores.

## Review Focus

- A genuine but irrelevant source must not close a content check.
- Multiple checks may have only one resolved: do not report all as complete.
- A downstream consumer without an OCR import is not automatically a missing SDK file.
- Old runtime results must not clear risks after application files or report change.
- Passing human-submitted logs do not authenticate execution or approve an upgrade automatically.

## Task 1: Evidence coverage

Files: change-review-agent/app/paddleocr_investigation.py, change-review-agent/tests/test_internal_evidence_coverage.py.
Produces: check assessments per version, unresolved content requirements, canonical check IDs preserved across replanning.
- [x] RED: irrelevant validated candidate remains unresolved; supported requirements resolve; unknown rule stays candidate; planner child resolution closes only its parent/version.
- [x] GREEN: explicit rule-derived fact groups, source validation before coverage, bounded continuation and coverage trace.
- [x] Verify: python -m pytest tests/test_internal_evidence_coverage.py tests/test_paddleocr_investigation.py.

## Task 2: Application roles and regressions

Files: versioned-rag-service/src/paddleocr_application_roles.py, src/paddleocr_compatibility.py, tests/test_internal_application_roles.py; demo-ui/services/paddleocr_application_inputs.py.
Produces: role inventory, appropriate configuration/contract gaps, risk-specific regression requirements.
- [x] RED: consumer/docs do not gain no-usage/configuration gaps; malformed configuration remains a gap; whole unrelated input remains unverified.
- [x] GREEN: classify without execution, attach roles, add contract tests to original bundle within 12-file limit.
- [x] Verify: python -m pytest tests/test_internal_application_roles.py tests/test_paddleocr_compatibility.py tests/test_paddleocr_application_graph.py.

## Task 3: Human disposition

Files: demo-ui/services/paddleocr_disposition.py, components/paddleocr_disposition.py, public_workbench.py, tests/test_paddleocr_disposition.py.
Produces: report-bound state, risk dispositions, externally submitted regression records, blocked/pending/recorded human decision.
- [x] RED: unmatched hashes, missing risk handling, unresolved gaps and missing tests prevent a positive decision; human logs remain unverified.
- [x] GREEN: bounded record validation, append-only audit snapshots, input/report invalidation, UI editor and export.
- [x] Verify: python -m pytest tests/test_paddleocr_disposition.py tests/test_review_audit.py tests/test_paddleocr_workbench.py.

## Task 4: Knowledge and source alignment

Files: demo-ui/services/internal_application_scope.py, public_workbench.py, tests/test_internal_application_scope.py; approved corpus selection/builder and source audit.
Produces: current application versus upgrade target selector, honest proxy labels, official setup/API-test source coverage.
- [x] RED: query defaults to current application dependency; explicit user version preserved; unsupported versions not offered.
- [x] GREEN: scope selector and clear result invalidation; inspect and import only relevant pinned setup/API-test files, record hashes and task reasons.
- [x] Verify UI and corpus suites; no previous metrics shown after data drift.

## Task 5: Bounded new validation and delivery

Files: evaluation/internal_workflow_v1/, README.md, docs/internal-workflow-release-2026-10-07.md.
- [x] Freeze fresh source-grounded questions and grouped application cases; preserve opened legacy holdouts.
- [x] Compare structural/window representations and scope BM25/available rerank paths; never count missing model paths as executed candidates.
- [x] Report retrieval fact coverage separately from generated-answer/impact accuracy; no fabricated human blind review.
- [x] Run service, Agent and UI test suites, independent review, fix important findings, and record remaining limitations.

Execution note: User selected Native previously and requested implementation now; proceed without another authorization cycle. Work stays in the current feature branch; this turn does not merge or deploy. Existing dirty files preclude broad staging or snapshot commits; leave changes reviewable and record test evidence in the ledger.
