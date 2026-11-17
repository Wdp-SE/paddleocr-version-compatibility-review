# 原创文档处理应用

用户为维护扫描文档处理流程的软件工程师。输入扫描页，经 OCR 调用、结果归一化、统一 JSON 输出，由下游消费文本。升级审查核对调用入口、返回结构、转换层和消费契约，输出须人工确认的风险和回归要求。

这些应用资料由项目作者编写，用于模拟真实维护任务，不是企业客户数据或 PaddleOCR 官方资料。官方接口依据来自固定提交中文文档和 API 实现。

本地离线运行只运行作者控制的应用；上传或粘贴的应用仅作 AST 文本分析。结果中的 OCR confidence 是识别器分数，不是 RAG 正确率。

## 显式配置加载

`pipeline.process_configured(path, config_path=...)` 调用显式 `ocr_client.create_configured_engine`，读取开发者指定的 UTF-8 JSON 文件（最多 16 KiB）。仅接受 `schema_version: 1` 和 `drop_score`；省略字段时阈值为 0.5，语言固定为中文。`drop_score` 必须是 [0,1] 内有限数值，布尔值、字符串、重复字段和未知字段均拒绝。没有配置参数时保留原始 `PaddleOCR(lang='ch')` 调用。

显式 `drop_score=` 覆盖文件值，记录为 `explicit_override`，不能把文件变化当作已被采用。每次调用重新加载文件，不使用历史配置缓存。可选 `provenance` 字典记录配置文件路径、原始字节 SHA-256、声明值、有效值、值来源和传入构造器的参数。构造器可由开发测试或固定本地 runner 注入；不允许从 JSON 导入代码。此配置分支只验证 PaddleOCR 2.9.1 的 `drop_score`，不扩展为任意版本参数迁移。

配置加载、分数边界夹具和真实模型推理分别记录。提高阈值不等于识别精度提升；只有样本输出与预先声明的验收目标才能说明实际行为。下游 `consumer.py` 不随此配置入口修改。

默认 `create_engine()`、`recognize(path)` 和 `process(path)` 保持无条件的旧版调用链，供升级审查定位；显式配置入口为 `create_configured_engine`、`recognize_configured` 和 `process_configured`。配置入口的动态生效由受控运行验证，不从默认入口静态结论推断。

## 显式升级目标入口

`pipeline.process_target` 和 `process_target_configured` 复用相同 OCR 调用层，但经已有 `normalize_target` 读取 PaddleOCR 3.0.0 的 `rec_texts` / `rec_scores`，再交给原 `consumer.to_json`。默认 `process` / `process_configured` 继续使用旧适配器，作为可复现的升级前审查基线。选择目标入口是开发者的明确操作，不按结果形状自动猜测版本。

此目标分支仅验证默认无阈值覆盖的 3.0.0 升级；旧版 JSON `drop_score` 不能据此声称已迁移。PaddleOCR 3.0.0 仍保留 `ocr` 调用别名；此处解决实际返回结构变化，不因别名存在推断结果兼容。目标入口须通过真实多页、空页、分数对应和下游 JSON 检查后才可记录运行通过。
