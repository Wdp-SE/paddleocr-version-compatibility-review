# 内部 OCR 组件维护工作台

面向公司内部研发、测试与应用负责人。人工知识查询和两条变更审查流程共用版本化证据服务。学生原型使用 PaddleOCR 官方公开资料与作者原创应用模拟维护任务，保留实际来源身份。

## 当前流程

- 知识查询：选择底层依赖版本，检索原文、生成带引用回答并保留缺口。
- 依赖升级：围绕同一登记应用的 PaddleOCR v2.9.1 → v3.0.0，核对调用、适配、输出与内部使用位置。
- 配置变更：依赖保持 v2.9.1，检查已支持的阈值变化、来源及回归要求。

主案例为扫描页 OCR → 归一化 → JSON。PPStructure 仅为相关静态检查分支，不代表默认应用已完成该功能运行验证。活动资料数量、来源和策略以 `/public/workspace` 的当前构建为准，不沿用旧版固定数字。

三条流程由 LangGraph 编排；任务完成、证据支持、实际运行和人工处置分别显示。Agent 不执行提交代码、不自动修改或批准发布。

## 本地运行

从项目根目录安装统一环境并启动后端与界面：

```powershell
python -m venv .venv-maintenance
.\.venv-maintenance\Scripts\python.exe -m pip install -r requirements-dev.txt
.\start_prototype.ps1 -RagPort 8775 -UiPort 8533
```

模型密钥由后端环境变量配置。仅检索时使用启动参数 `-DisableGeneration`。已单独启动 API 时，可以从本目录仅启动界面：

```powershell
$env:APP_ENV = 'public_demo'
$env:RAG_API_BASE_URL = 'http://127.0.0.1:8775'
..\.venv-maintenance\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8533
```

`start_demo.ps1` 同样使用项目根目录的统一环境，不再回退旧工程目录。公开演示只处理公开与原创资料；内部部署须配置并验收身份、权限、持久任务与可信运行记录。

## 相关说明

- [业务边界](../docs/business_scope.md)
- [两类维护流程](../docs/change_review_workflows.md)
- [LangGraph 迁移与验证](../docs/maintenance/2026-11-12-langgraph-migration.md)
- [内部部署](../deploy/internal/README.md)

公网部署、真实 OCR 回归、人工回答正确性和生产基础设施分别验收；界面或单元测试通过不代表这些条件已满足。
