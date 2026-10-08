# Production readiness and public demo verification

## Current public prototype

The active product profile is `pphuman`: versioned Chinese engineering material from the official PaddlePaddle/PaddleDetection repository, covering PP-Human pedestrian detection, tracking, attributes, behavior analysis, inference configuration, and deployment. Six release snapshots from v2.5.0 to v2.9.0 are indexed; latest defaults to v2.9.0. The workbench supports release-scoped retrieval and human-reviewed change-impact candidates. It does not connect to private tickets, video/test results, or approval systems, and it does not write back to upstream sources. Other public corpus snapshots were removed from the service package; public-demo startup is pinned to this corpus.

- `/health` and `/public/workspace` expose separate source-corpus, retrieval-configuration, evaluation, and build fingerprints. The public workspace rejects an unexpected corpus/profile/language rather than silently serving it.
- The Streamlit “系统说明” page reports the frontend and RAG revisions separately.
- API responses include `X-Request-ID`; safe diagnostics record route, status, stage, provider/model label and timing without logging raw queries, prompts, response bodies, or credentials.
- Explicit requests for private company test results and approvals are rejected before retrieval or generation. The release smoke verifies this with a fixed no-model-call probe.
- The public workbench is unauthenticated and uses public material. Its session review history is not a durable enterprise audit trail.

## Read-only post-deploy smoke

After both hosted services have deployed the intended GitHub `main` commit, copy the frontend SHA from **系统说明 → 运行版本与资料指纹** and run this command from the repository root:

```powershell
python versioned-rag-service/scripts/public_release_smoke.py `
  --ui-url https://paddleocr-version-compatibility-review.streamlit.app/ `
  --ui-revision <40-character-frontend-sha> `
  --api-url https://version-aware-rag-public-demo.onrender.com `
  --expected-sha <40-character-github-main-sha>
```

The smoke checks UI availability, matching frontend/API/Git revisions, matching source/config/evaluation fingerprints, the Chinese-only PP-Human workspace, two public retrieval probes against v2.9.0, and a private-information refusal. It does not call a model for these probes. Reported service timings are remote request timings, not a long-term availability or latency guarantee.

Exit code 0 means those checks passed at that moment. Unknown Git metadata, mismatched revisions, stale evaluation fingerprints, an unexpected corpus, or a failed scope probe must remain a failure; do not describe the site as updated when the check fails.

## Current evaluation evidence and limits

The PP-Human corpus is a new evaluation target. Its benchmark is pending; prior scores from unrelated corpora must not be presented as PP-Human retrieval or agent performance. Current checks confirm source/version filtering, evidence binding, and the human-review boundary, but do not establish retrieval recall, answer factuality, impact-candidate precision, or hallucination rate. Image references in the source documents are not OCR-indexed.

Once a frozen PP-Human evaluation set is available, publish its query count, source-family holdout split, required-source recall, version correctness, answer citation validity, and change-impact candidate precision/recall together. Do not report latency or quality as a validated metric before that run completes.

## Future private deployment work

Before indexing private engineering material, the design still needs authenticated identity and role mapping, tenant isolation, ACL filtering before retrieval, durable and encrypted audit storage, retention/deletion rules, approved model routing and regional policy, abuse/cost limits, operational alerting, and legal/security approval. The public prototype does not implement or claim these controls.
