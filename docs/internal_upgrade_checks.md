# 内部 OCR 组件升级检查

目标是公司内部维护扫描文档应用的研发、测试和应用负责人。PaddleOCR 官方资料代理内部组件知识；原创应用和契约保留学生 POC 身份。审查在升级上线前进行，不维护对外 SDK 客户支持。

## 检查与回归

1. 选择登记应用和不可变提交；候选模式另选基准提交。两个提交均收集完整登记范围，包括未修改的结果适配层和消费者。
2. 先查询本次应用契约与测试定义，再静态追踪调用、参数和结果读取。组件事实只来自固定的两版官方资料。
3. 静态发现必须绑定代码位置与官方版本证据。应用字段、分页策略来自应用原文；测试定义中的旧断言与生产风险分级展示。
4. 当前/目标/参考依赖文件由应用契约声明；角色冲突或未知须补齐，不能默认为当前环境。
5. 修改后选择新提交复查。风险消失不等于修复成功；回归记录须匹配该提交、文件、报告、依赖环境和契约要求。

跨函数参数分析有六层、200 个调用上下文的上限。嵌套函数、动态调用、复杂返回路径及图关系未证明部分继续留痕。一般语法限制不逐项变成业务风险；未解决的实际绑定、资料及环境缺口仍保留。Agent 不执行提交代码或批准升级。

## 只读 Git/CI 预检查

```powershell
python versioned-rag-service/scripts/review_registered_upgrade.py --head HEAD --output .local-demo-temp/upgrade-review.json
```

`--base <本地提交>` 可同时检查基准快照。只接受已登记仓库中的本地 Git 引用，不拉取任意地址。退出码 2 表示静态风险，3 表示仍须人工/运行核验；静态无风险不会返回自动批准。`--advisory` 将发现作为建议记录并返回 0，不代表通过升级验收。

`.github/workflows/internal-ocr-review.yml` 使用只读权限、固定 Actions 提交和无凭据持久化的 checkout。它验证静态边界、原创入口夹具及生成建议报告，不调用模型、不运行 OCR 模型、不写 PR 评论、不接触企业资料。默认旧适配器保留为升级风险示例，因此 CI 建议报告不能当作应用发布门禁。

## 原创应用的运行验证

```powershell
python -m pytest examples/paddleocr_document_app/application/tests/test_contract.py -q
python examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/ocr-fixtures --output .local-demo-temp/ocr-runtime.json
```

第二条命令仅在开发者控制的对应 PaddleOCR 环境中运行作者应用。输入是既有原创合成扫描页。脚本使用显式本地模型配置，通过实际调用封装进入 `application.pipeline.process`，直接检查输出字段、页数/页号、预期文字和空页。`entrypoint_regression` 与单独的适配器探针分开；入口失败不会回退为适配器成功。工作区有修改时明确记录 dirty，不能将结果标成已提交版本验证。

夹具注入、真实 OCR 推理、版本证据检索和人工上传日志属于不同证据级别。新版字段适配的夹具通过不代表已提交的升级分支或真实环境通过；人工日志仍未被服务认证。

## 检索专项回归

```powershell
python evaluation/internal_upgrade_depth_v1/run_checks.py --output .local-demo-temp/upgrade-task-checks.json
```

该已知任务集检查双版本说明、调用参数、官方测试、应用契约和应用测试的原文覆盖，同时检查未提供的内部审批事实不被官方资料替代。不调用模型，不输出回答准确率；代码、原文和应用文件哈希绑定结果。不能把这些已见题的通过率当成盲测或企业收益。
