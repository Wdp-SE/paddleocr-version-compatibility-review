# 原创文档处理应用静态审查样例

三个 Python 文件是学生自行编写的扫描文档适配器片段，分别展示旧版版面入口、旧版 OCR 嵌套结果消费，以及保留的基础调用表面。它们用于受限文本静态审查，不是上游示例副本。

将文件内容粘贴到兼容性审查入口，选择当前依赖 `v2.9.1` 和目标依赖 `v3.0.0`：

| 文件 | 预期静态结论 |
| --- | --- |
| `removed_structure_api.py` | 官方升级说明支持 `PPStructure` 被移除的风险；人工核对 `PPStructureV3` 与版面归一化契约 |
| `legacy_result_consumer.py` | 两版官方代码与结果示例支持嵌套位置消费的迁移风险 |
| `compatible_basic.py` | 官方材料支持 `PaddleOCR(lang="ch").ocr(input)` 的调用表面；未证明实际推理或结果契约兼容 |

审查器不执行用户输入，不读取传入的 `scan_path`，不修改应用代码。静态报告中的 `runtime_verified=false` 始终表示用户应用尚未运行验证。

## 原创应用真实升级回归

`run_upgrade_demo.py` 只运行本目录的原创应用：对 `fixtures/` 的两页中文样例和一页空白页进行真实 OCR，先尝试旧结果消费代码，再用 `result_adapter.py` 转成统一的 `{text, confidence, page_index}` 契约。图片为自行生成的测试输入，不是客户扫描件。

历史实测记录在 `evaluation/paddleocr_quality_v2/runtime-v2.json` 和 `runtime-v3.json`：旧消费代码在 2.9.1 可运行，在 3.0.0 遇到 `IndexError`；当时版本适配器的三页回归通过。这些历史报告不验证当前应用源码。这里验证的是结果消费与适配，不是识别准确率或任意应用兼容性。旧版使用 PP-OCRv4、新版使用 PP-OCRv5，不能据此比较 OCR 质量或速度。

在两个独立 Python 环境分别安装 PaddleOCR 2.9.1 / PaddlePaddle 2.6.2，以及 PaddleOCR 3.0.0 / PaddlePaddle 3.0.0 / PaddleX 3.0.0。此次验证环境继承了本机库，并非完整干净依赖镜像，报告记录实际版本。模型需要提前下载到本地；不能把模型资产和推理环境当成公网演示已安装的依赖。

```powershell
# 若需要重新生成输入，显式选择可用的中文字体；默认 Windows 微软雅黑。
$env:OCR_DEMO_FONT='C:/Windows/Fonts/msyh.ttc'
python examples/paddleocr_document_app/run_upgrade_demo.py --create-fixtures --fixtures .local-demo-temp/ocr-fixtures
# 在对应的隔离环境中执行；版本不符时 runner 拒绝运行。
python examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/ocr-fixtures --output .local-demo-temp/runtime.json
```

模型目录可通过 `OCR_V2_MODEL_ROOT` / `OCR_V3_MODEL_ROOT` 指定；默认分别为当前用户的 `.paddleocr/whl` 和 `.paddlex/official_models`。Windows 中文路径上的新版图像读取故障通过本地解码为 BGR 数组解决。空页返回空结果、异常字段或非有限分数不能静默当作成功。

## 显式 JSON 配置与失败记录

`application.pipeline.process_configured` 支持 `config_path`，由正常应用工厂加载 UTF-8 JSON（最多 16 KiB）：`{"schema_version":1,"drop_score":0.8}`。仅支持 2.9.1 阈值分支；未知/重复字段、错误类型、非有限值和越界值均拒绝。默认阈值 0.5；显式 `drop_score` 高于文件值的优先级，并单独记录来源。无配置时保留原调用行为。配置字节哈希、有效值和构造器参数随记录保存，静态配置追踪不等于此运行路径已执行。

```powershell
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/runtime_acceptance.py --output .local-demo-temp/business-quality-20261114/runtime-fixtures.json
# 在固定 2.9.1 环境运行；仅使用提前准备好的本地模型。
.local-demo-temp/ocr-v2/Scripts/python.exe examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/business-quality-20261114/ocr-fixtures --config .local-demo-temp/business-quality-20261114/runtime-configs/new.json --output .local-demo-temp/business-quality-20261114/real-v2-json-new.json
```

固定分数 runner 验证阈值下方/相等/上方、页序、空页、字段类型、文字与置信度对应，并保留“文件更新但覆盖值未变”“适配器成功但应用入口失败”等反例。它不执行待审应用，不接受代码路径参数。升级候选适配器仅在明确标注的夹具中注入，不代表默认应用已修改或真实升级已通过。

真实 OCR runner 检查实际应用入口和独立适配器探测，入口失败仍记录 `FAILED` 并返回非零退出码。异常也写失败报告，不能用适配器通过覆盖失败。当前默认应用仍使用旧结果归一化，3.0.0 的真实入口必须单独验证。当前重验结果见 `evaluation/business_quality_v1/runtime-acceptance.md`，历史报告仅表示当时源码与环境。

默认 `create_engine()`、`recognize(path)` 和 `process(path)` 保持无条件的旧版调用链，供升级审查定位；显式配置入口为 `create_configured_engine`、`recognize_configured` 和 `process_configured`。配置入口的动态生效由受控运行验证，不从默认入口静态结论推断。

## 选择显式 3.0.0 目标适配器

`application.pipeline.process_target` / `process_target_configured` 使用已有 `normalize_target`，将 `rec_texts` / `rec_scores` 转为同一消费者的输出。默认入口仍是升级前基线，目标入口不自动代替默认入口。此分支的 3.0.0 验收只用默认配置；没有证明旧 `drop_score` JSON 参数完成迁移。

`evaluation/business_quality_v1/v3_application_acceptance.py` 只运行本原创应用，检查真实旧入口失败、目标入口三页输出、空页、文字与分数对应、消费者未变及源码身份。固定使用 PaddleOCR / PaddleX / PaddlePaddle 3.0.0 和本地 PP-OCRv5_server 检测/识别权重；CPU 配置显式限制检测最长边为 960，关闭三个辅助模型和 MKL-DNN，记录全部参数及权重哈希。此运行配置不能等同上游默认配置，也不证明 PPStructure 分支。

```powershell
.local-demo-temp/ocr-v3/Scripts/python.exe evaluation/business_quality_v1/v3_application_acceptance.py --output .local-demo-temp/resolve-20261114/v3-target-application.json
```
