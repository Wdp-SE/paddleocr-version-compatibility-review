# PaddleOCR RAG quality V4 execution ledger

Approved scope: keep the PaddleOCR corpus and application upgrade review business;
prefer evidence correctness over millisecond latency. Native inline implementation
on `codex/paddleocr-rag-quality-v4` preserves the current checkout. Existing PP-Human
changes are unrelated and will not be included.

1. Repair product/API/module scope interpretation without relaxing version checks.
2. Assemble bounded canonical table/code context and cover compound requirements.
3. Compare BM25 windows, neural reranking and constrained model ranking using
   source-grounded development questions; freeze new holdout before selection.
4. Add at most one corrective retrieval after content verification failure, retain
   verified claims, and expose concrete unresolved requirements and call diagnostics.

Validation: meaningful scope/context/retry regressions, current source-grounded
retrieval and live generation probes, related service/Agent/UI regression tests.
V3 sealed labels are already opened and are not a new blind test. Do not claim
retrieval hits are answer correctness or static Agent success is RAG uplift.

Progress: baseline defects identified in query_module and isolated evidence spans.

Completed locally: explicit product/API/pipeline scope repair; canonical bounded
table/code assembly separate from exact-span retrieval metrics; pre-truncation
scope filtering and compound requirement coverage; constrained ID-only ranking;
one content correction with per-pass attribution and reported total usage.
Fresh review found and fixed owner precedence, table scope, oversized long-line
context, same-source span union, verification indexing and missing report locks.

Current frozen development comparison (12 questions): legacy 9, scoped BM25 10,
MiniLM 10, Chinese BGE 11, RRF+BGE 10, constrained model ranking 11 complete.
Select model ranking by development coverage then non-target-module fragments
(5 vs BGE 6). Grouped holdout (8): legacy 5, scoped BM25 7, selected ranking 7,
both scoped configurations 0 wrong-module and 0 wrong-version fragments.
Ranking did not improve holdout beyond scoped BM25; superiority is not established.
No holdout retuning. Promotion fallback rule was added before reading final
candidate results, although the file was already generated; disclose this detail.

Five actual DeepSeek probes: four supported topic answers and one evidence-bounded
negative guarantee answer. 17.6–40.5 seconds, 67119 provider-reported total tokens,
one correction. These are operational/source-audit probes, not population accuracy.
Remaining exact fact misses: v4-03 code-use example, v4-14 angle enumeration.

Validation: final active-release service 441 passed/1 skipped; related Agent 38 passed; UI 173 passed/
10 skipped. UI sandbox temporary-directory failures were rerun with normal
Windows temporary access and passed. After final auto-policy integration, focused
service release/API suite 36 passed; final UI diagnostics 7 passed. PyPDF2 warns
about deprecation. No public deployment, commit, push or merge this iteration.
Local fresh processes: RAG 8774, UI 8523; workspace and automatic ranking checked
against this release. Release hash lock validated with canonical metric recompute.
