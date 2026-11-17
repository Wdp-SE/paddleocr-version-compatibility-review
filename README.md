# 内部 OCR 组件维护工作台

面向内部研发、测试和应用负责人的个人项目。用 PaddleOCR 官方公开资料代理版本化组件知识，用原创扫描文档应用模拟 OCR 调用、结果归一化和 JSON 消费。

[在线体验](https://paddleocr-version-compatibility-review.streamlit.app/) · [业务范围](docs/business_scope.md) · [工作流](docs/change_review_workflows.md)

## 三个可复现的案例

| 任务 | 操作 | 核对结果 |
| --- | --- | --- |
| 版本查证 | 在知识查询中选 v3.0.0，查询 `rec_texts` 和 `rec_scores`，打开原文 | 查看来源版本、出处及行号 |
| 依赖升级 | 在变更审查中运行旧结果消费案例，再运行字段适配案例 | 对照 v2.9.1 → v3.0.0 的风险位置、两版依据与回归要求 |
| 配置审查 | 在同一应用和 v2.9.1 下对照识别分数阈值 | 查看值的声明和覆盖路径、空页及 JSON 契约的验证要求 |

RAG 采用版本前置过滤、结构化 BM25、多查询排序融合和逐项引用核验。Agent 用 LangGraph 编排静态检查、定向查证和报告。内部工作台可按登记应用的 Git 提交及入口查证，报告按项关联受信 CI 记录。静态风险、影响候选、夹具、实际 OCR 运行和人工处置分别保留来源。

公网入口使用公开和原创示例。实际回归由开发或受控 CI 环境执行，应用负责人通过组织流程作出处置。企业身份、PostgreSQL 及真实 CI 的接入需单独验收。 [内部部署说明](deploy/internal/README.md)

## 本地运行

使用 Python 3.11 或 3.12。在 Windows 仓库根目录运行以下命令，可在无模型密钥环境体验证据检索和固定静态案例。

```powershell
python -m venv .venv-maintenance
.\.venv-maintenance\Scripts\python.exe -m pip install -r requirements-dev.txt
.\start_prototype.ps1 -DisableGeneration
```

模型生成可通过服务端配置启用。分开启动、端口及网络设置见[界面运行说明](demo-ui/README.md)。

## 验证

- 离线检索验收包含 22 道固定题和 9 道开发改写，逐项核对必要原文事实、版本、模块和证据不足时的行为。 [数据与复现命令](evaluation/business_quality_v1/README_retrieval.md)
- 原创应用用同一组三页样本比较旧入口和目标适配入口，覆盖真实推理、结果归一化、空页及同一 JSON 消费者。记录附依赖、样本、权重和应用文件身份。 [运行与限制](evaluation/business_quality_v1/v3-resolution.md)
- GitHub Actions 运行静态、夹具、内部 API 和检索回归，保留逐项结果。历史试验按各自构建和题集解释。 [历史评测索引](evaluation/archive/README.md)

## 代码入口

- `versioned-rag-service/` 提供 FastAPI 证据服务、应用提交查证、任务和结果记录。
- `change-review-agent/` 编排升级与配置审查，生成证据和回归要求。
- `demo-ui/` 提供公开演示及受身份保护的内部工作台。
- `examples/paddleocr_document_app/` 包含原创 OCR 应用、输出契约和回归入口。
