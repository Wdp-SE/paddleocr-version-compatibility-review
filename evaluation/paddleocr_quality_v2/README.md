# PaddleOCR 质量选型与升级验证

面向维护文档处理应用的研发人员，按依赖版本查询官方资料，并对 2.9.1 → 3.0.0 升级做受控兼容性审查。准确与可靠优先，延迟只作资源代价记录；不将重排分数解释为正确概率。

## 冻结对比

`cases.json` 包含 12 个开发集、12 个内部留出集问题和 4 个无答案探针。每个可回答问题以固定版本、原文路径及事实 token 组合标注。同版本同语料 Top-5 对比 BM25、BGE 中文语义召回、BM25 + 多语言 cross-encoder、混合 RRF、混合 RRF + cross-encoder。候选召回各 20 条，混合后最多 40 条，重排取最终 Top-K。

只按开发集的完整事实召回、必需事实召回决定候选方案；留出集不参与选型。题目由开发者按原文标注，不是业务专家独立盲测。报告绑定题集、语料、runner、检索及指标代码，构建不一致时页面隐藏它。事实 token 命中属于检索覆盖指标，不能当成答案准确率、业务影响召回或生产最优证明。

配置已下载的本地模型后运行：

```powershell
$env:PADDLEOCR_EMBEDDING_PATH='<BAAI/bge-small-zh-v1.5 本地 snapshot>'
$env:PADDLEOCR_RERANKER_PATH='<cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 本地 snapshot>'
$env:PADDLEOCR_VECTOR_CACHE='.local-demo-temp/quality-vectors'
python evaluation/paddleocr_quality_v2/run_evaluation.py
```

语义模型使用真实神经 Embedding，不是字符哈希向量。CPU eager attention 用于规避此次 Torch 2.4 / Transformers 5.4 的 SDPA 非有限输出；模型异常时明确回退 BM25，保留原因码，不返回“重排成功”。Cross-encoder 最多读取 512 tokens，长块尾部信息可能丢失，这是本轮的已知限制。

参考：[BGE 官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[多语言重排模型说明](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)。本轮固定上述各一个模型，只能比较这些候选组合。

## 生成与审查边界

生成先校验引用 ID，再以引用原文逐条进行模型支持度复核。失败或无支持的主张不显示；部分支持时保留通过部分并显式列出缺口。复核最多一次模型请求、至多五条主张、上下文预算 60,000 字符；失败仍保留原始检索证据。生成与复核分开记录模型、usage 和时延，金额未知时不虚构成本。

复核使用同一基础模型，可能有相关误差，并非独立人审或幻觉率证明。`online-report.json` 是实际 DeepSeek 公共资料冒烟记录，不是人工准确率评测。业务审查的确定性发现由静态工具和固定官方版本证据支持，模型补充建议不能改变已验证发现、执行用户代码或签署升级通过。

`runtime-v2.json` / `runtime-v3.json` 只证明本仓库原创三页样例的旧结果消费者失败和版本适配器成功；`uploaded_application_runtime_verified` 保持 false。不同 OCR 模型和受其他任务干扰的本机计时不能作为 OCR 性能对比。

独立审查者另行编写的 probes 只在方案冻结后运行，不针对它们改词典、提示词或参数；即使通过，也不能代替更多真实应用的人工盲审。
