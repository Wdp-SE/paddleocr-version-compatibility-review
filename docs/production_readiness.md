# Production readiness and public demo verification

## Current public prototype

The active product profile is `paddleocr`: fixed Chinese documentation, Python interfaces and YAML configuration from PaddlePaddle/PaddleOCR v2.9.1 and v3.0.0. Public material represents internal component knowledge in this student POC; the original scan-page OCR application supplies its own contract and test definitions. The product serves internal developers, testers and application owners, without claiming real enterprise users or production data. Public-demo startup is pinned to this corpus.

- `/health` and `/public/workspace` expose separate source-corpus, retrieval-configuration, evaluation, and build fingerprints. The public workspace rejects an unexpected corpus/profile/language rather than silently serving it.
- The Streamlit “系统说明” page reports the frontend and RAG revisions separately.
- API responses include `X-Request-ID`; safe diagnostics record route, status, stage, provider/model label and timing without logging raw queries, prompts, response bodies, or credentials.
- Explicit requests for private company test results and approvals are rejected before retrieval or generation. The release smoke verifies this with a fixed no-model-call probe.
- The registered-application input reads an allowlisted local Git snapshot, including unchanged consumers, and keeps application commits separate from SDK releases. It does not clone arbitrary repositories, execute submitted code, or approve upgrades.
- Application contracts and tests are searched through `/public/application-search` in a separate request scope, not merged into official SDK evidence. They do not prove that tests have run.
- The public workbench is unauthenticated and uses public or student-original material. Local SQLite task records and recovery credentials are not an enterprise audit or identity system, and ephemeral hosting disks may lose them.

## Read-only post-deploy smoke

After the backend has deployed the intended GitHub `main` commit, run this command from the repository root. Separately read the frontend SHA from **系统说明 → 运行版本与资料指纹** and compare it with that commit; HTTP 200 alone cannot verify UI revision.

```powershell
python versioned-rag-service/scripts/paddleocr_release_smoke.py `
  --ui-url https://paddleocr-version-compatibility-review.streamlit.app/ `
  --base-url https://version-aware-rag-public-demo.onrender.com `
  --expected-revision <40-character-github-main-sha> `
  --output .local-demo-temp/deploy/verified.json
```

The smoke checks backend revision and readiness, the Chinese PaddleOCR publisher scope, source counts, version-filtered retrieval against both releases and UI HTTP availability. It does not call a model. Inspect the live Git input separately: the current application commit should collect the registered files, and the new application-search API should preserve that commit and original-file hashes. Reported service timings are point-in-time checks, not availability or latency guarantees.

Exit code 0 means the backend checks passed at that moment; the smoke explicitly records UI revision as unverified by HTTP. Unknown frontend Git metadata, mismatched revisions or unexpected corpus must be reported rather than presented as a fully verified deployment. Historical evaluation records must not be re-labelled as current after code or corpus changes.

## Current evaluation evidence and limits

Current known-question diagnostics and remaining failures are recorded in `evaluation/reliability_v1/README.md`; Git review acceptance is in `evaluation/internal_git_review_v1/README.md`. Literal evidence coverage, model support checks and static risk detection are separate measures. They do not establish independent answer accuracy, complete application impact analysis or actual OCR runtime compatibility. Image references in source documents are not OCR-indexed.

New strategy selection requires matching source, code and evaluation identities. Do not reuse an older six-strategy comparison as evidence for the changed implementation, or claim that absence of a static risk proves an upgrade safe.

## Future private deployment work

Before indexing private engineering material, the design still needs authenticated identity and role mapping, tenant isolation, ACL filtering before retrieval, durable and encrypted audit storage, retention/deletion rules, approved model routing and regional policy, abuse/cost limits, operational alerting, and legal/security approval. The public prototype does not implement or claim these controls.
