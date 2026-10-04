# PaddleOCR 应用依赖升级审查 Agent

当前公开工作台使用 app/paddleocr_review.py 的 PaddleOCRReviewAgent。业务聚焦维护扫描文档 OCR、版面/表格解析及结果归一化应用，检查 PaddleOCR v2.9.1 → v3.0.0 对应用调用和旧结果消费的影响。RAG 同时服务人工查资料和 Agent 查证据，两者不是独立项目。

## 当前受控流程

受限应用文本 → 固定版本及知识空间校验 → 只读 AST 兼容性工具 → 应用行号/摘录/文件哈希校验 → 官方旧/新版本证据身份校验 → 同空间 RAG 相关资料查询 → 可选模型核查建议 → 影响项与验证清单 → 人工审阅及 JSON 审计。

静态工具不执行或导入提交代码。已支持规则包括 PPStructure 移除和旧 OCR 嵌套结果消费；基本调用形式未发现变更只表示该狭窄 API 表面有证据支持。动态或不明确绑定、未覆盖参数、配置和环境进入缺口，不给出已兼容结论。工具发现要求应用位置与两版固定提交、来源哈希、原文行号绑定；相关检索不能自行增加已确认风险。

可选模型建议最多发起一次网关请求，传入不超过8个已核验目标版证据及不超过4000字的核查摘要。超长摘要明确截断，完整静态报告始终保留。模型建议必须绑定允许的证据，并标为未验证；生成失败不妨碍静态审查。摘要及证据可能发往后端配置的模型服务，请仅使用公开或已获授权的应用资料。

## 验证与边界

三个本项目原创样例位于 examples/paddleocr_document_app，分别覆盖移除接口、旧结果消费者、基本调用形式。固定官方中文与接口/配置证据共54条来源；专项验收见 evaluation/paddleocr_compatibility_v1。静态规则通过及十条来源派生查询不等于独立业务准确率，也不证明实际 OCR 推理已通过。

报告始终 runtime_verified=false。开发者仍需验证真实扫描件、模型加载、空页、多页、版面/表格质量和下游契约。人审只记录是否已审阅，不自动修改应用、公共基线或上游资料。会话隔离的 SQLite 和 JSON 可追溯记录是匿名演示能力，不是实名审批或可靠云持久存储。

## 历史可复用组件

旧领域审查和 Word 文档工作流保留为可复用组件及回归夹具，不属于当前 PaddleOCR 活动默认演示。以下为该历史工作流的使用说明。
下方“业务流程”说明的是可复用的结构化 Word 文档工作流子系统；它与公开工作台的假设变更审查入口相邻，但不是对外宣称的自动文档发布流程。

该可复用子系统面向一个业务问题：用户选择项目、文档和版本范围，上传结构化
Word 模板，系统在限定 RAG Scope 内收集 Evidence，生成证据约束草稿，
经单审核人逐章节审核后输出正式文档。它不是网页研究、知识采集、通用
聊天或万能办公 Agent。

## 业务流程

```text
WorkflowScope
  → DOCX Template Parse
  → SectionTask / FieldTask
  → bounded RetrievalQuery
  → RAG POST /retrieve
  → Evidence Sufficiency
  → EXTRACTIVE / controlled GENERATIVE FieldDraft
  → Draft DOCX + Citation Appendix
  → Section Edit / Approve / Reject
  → Finalization Guard
  → approved.docx
```

Checkpoint 保存 Scope fingerprint、任务、草稿、审核和 Evidence。Resume
会校验 Scope 不变；文档版本变化时将 STALE Evidence 保留为审计记录，
只重新执行受影响 Section。

## 核心边界

- UI/CLI 只调用 `DocumentWorkflowFacade`。
- `HTTPRetrieveClient` 是唯一 RAG HTTP 边界；不直接访问 FAISS。
- `FieldDraftingService` 是唯一 Drafting 边界；默认 `EXTRACTIVE`。
- `WorkflowRepository` 与 `CheckpointStore` 是本地持久化边界。
- `ReviewService` 只实现 Single-reviewer MVP，不是 OA 审批系统。
- AI 不得自动批准章节；全部必要章节 APPROVED 后才能生成正式 Word。

## Word 模板范围

支持 Heading/标题 1–3、Outline Level 1–3、段落、简单表格，以及
`{{字段名}}` 或 `【待填写】` 占位符。替换跨 Run 占位符时保留未受影响
Run 的字体属性。文档末尾生成“引用来源”附录，列出文档、版本、状态、
章节、页码和 Evidence ID。

旗舰模板：
`project_delivery/document_workflow_business_refactor/demo/requirement_change_impact_template.docx`

## 快速验证

要求 Python 3.12：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\run_business_e2e.py
```

安全 E2E 完全离线，使用合成 Evidence，覆盖多 Evidence 草稿、MISSING、
编辑、批准、驳回、正式输出门禁和引用附录。

## CLI

```powershell
.venv\Scripts\python.exe run_document_workflow.py `
  --runtime-root runtime `
  --rag-url http://127.0.0.1:8765 `
  run `
  --template project_delivery\document_workflow_business_refactor\demo\requirement_change_impact_template.docx `
  --scope '{"project_ids":["DEMO-RD"],"active_only":true}'

.venv\Scripts\python.exe run_document_workflow.py --runtime-root runtime list
```

配置变量见 `config/document_workflow.example.env`。

## 数据安全

`GENERATIVE` 只允许 synthetic、public、approved-redacted 或明确配置的
本地模型。真实未经授权企业资料不得发送公网模型。`EXTRACTIVE` 是默认
模式；无 Evidence 返回 MISSING，证据不足返回 INSUFFICIENT_EVIDENCE，
未知或陈旧 Evidence 会阻止正式输出。

## 项目演进

项目早期基于 OpenManus 探索过 Knowledge Research，随后按企业研发文档
业务收敛为当前单一工作流。旧产品功能不再属于运行时；演进记录保留在
Git 历史中。

## 文档

- `docs/architecture.md`
- `docs/review_and_finalization.md`
- `docs/evidence_freshness.md`
- `docs/runbook.md`
- `docs/known_limitations.md`
- `project_delivery/document_workflow_business_refactor/legacy_removal_plan.md`
