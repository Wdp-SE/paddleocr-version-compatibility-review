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

实测记录在 `evaluation/paddleocr_quality_v2/runtime-v2.json` 和 `runtime-v3.json`：旧消费代码在 2.9.1 可运行，在 3.0.0 遇到 `IndexError`；版本适配器的三页回归均通过。这里验证的是结果消费与适配，不是识别准确率或任意应用兼容性。旧版使用 PP-OCRv4、新版使用 PP-OCRv5，不能据此比较 OCR 质量或速度。

在两个独立 Python 环境分别安装 PaddleOCR 2.9.1 / PaddlePaddle 2.6.2，以及 PaddleOCR 3.0.0 / PaddlePaddle 3.0.0 / PaddleX 3.0.0。此次验证环境继承了本机库，并非完整干净依赖镜像，报告记录实际版本。模型需要提前下载到本地；不能把模型资产和推理环境当成公网演示已安装的依赖。

```powershell
# 若需要重新生成输入，显式选择可用的中文字体；默认 Windows 微软雅黑。
$env:OCR_DEMO_FONT='C:/Windows/Fonts/msyh.ttc'
python examples/paddleocr_document_app/run_upgrade_demo.py --create-fixtures --fixtures .local-demo-temp/ocr-fixtures
# 在对应的隔离环境中执行；版本不符时 runner 拒绝运行。
python examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/ocr-fixtures --output .local-demo-temp/runtime.json
```

模型目录可通过 `OCR_V2_MODEL_ROOT` / `OCR_V3_MODEL_ROOT` 指定；默认分别为当前用户的 `.paddleocr/whl` 和 `.paddlex/official_models`。Windows 中文路径上的新版图像读取故障通过本地解码为 BGR 数组解决。空页返回空结果、异常字段或非有限分数不能静默当作成功。
