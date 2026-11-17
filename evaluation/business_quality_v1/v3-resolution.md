# 原创应用 PaddleOCR 3.0.0 运行问题闭环

2026-11-14。本记录仅针对作者控制的扫描页 OCR → 归一化 → 下游 JSON 原创应用。

已完成真实运行：升级前旧入口保留为失败基线；新增显式 `process_target` / `process_target_configured`，通过已有 `normalize_target` 转换新版结果，消费者未改。当前源码的真实目标入口、四条 2.9.1 配置路径及固定分数反例均已重验。结构化汇总见 [runtime-summary-resolution.json](runtime-summary-resolution.json)。

## 定位证据

历史原生退出 `0xC0000005` 未再次出现，不能声称已查明其具体原因。先分离识别器、模型初始化、检测推理、识别推理和应用适配边界，再重跑原 runner：

|实验|实际观察|
|---|---|
|最小服务端识别器|初始化约 0.47 秒；中文裁剪页真实识别约 1.51 秒。|
|原服务器模型单页管线|两个 `create_predictor` 均返回；MKL-DNN 实际为关闭；检测输入为 `1×3×736×1472`，检测 36.04 秒、识别 5.47 秒，输出预期文本。|
|原始 runner 与原导入顺序|228.04 秒完成六次检测调用及真实识别；三条旧应用入口均抛出 `ValueError`，独立适配器及样本检查通过。此记录来自新增目标入口前的源码，作为定位历史保存。|
|显式 CPU 检测最长边 960|输入为 `1×3×480×960`，检测 12.44 秒、识别 5.63 秒；实际文字满足同一页验收。|

后三项原始日志依次位于本地忽略目录 `.local-demo-temp/resolve-20261114/`：`v3-pipeline-baseline.log`、`v3-current-original-order.log`、`v3-max960-single.log`。最小识别器记录保留在本次工具原始输出中。原 runner 的领域状态为 `FAILED`；当时诊断包装器返回过零退出码，不能将包装器退出码当作应用通过。后续诊断包装器已在领域失败时返回非零。

可复现的应用错误是旧入口继续用 `normalize_v2` 读取新版返回对象。真实新版页是 `paddlex.inference.pipelines.ocr.result.OCRResult`，其中 `rec_texts` 和 `rec_scores` 均为列表；现有严格 `normalize_v3` 已接受该字典子类，无需放宽结果类型校验。

## 当前真实验收

`v3-target-application.child.json` 与监督记录 `v3-target-application.json`：**PASSED**，监督进程 82.578 秒、退出码 0、未超时、运行前后源码一致。旧 `process_configured` 对第一张真实页执行一次，保持预期 `ValueError: invalid v2 page`；目标 `process_target_configured` 接收按顺序排列的三张本地解码 BGR 数组，作为**一次应用调用**。不是把三次单页输出拼接冒充同次多页应用结果。

目标结果通过字段与类型、6 组文字/置信度对应、0/1/2 页序、末尾空白页、预期文本和同一消费者 JSON 检查。原始报告保存实际结果、分数、依赖版本、配置和全部模型权重哈希；独立适配器成功不覆盖旧入口失败。

当前 CPU 配置明确为：`device='cpu'`、`cpu_threads=2`、`enable_mkldnn=False`；关闭文档方向分类、文档矫正和文本行方向分类；`text_det_limit_type='max'`、`text_det_limit_side_len=960`、`text_rec_score_thresh=0.0`；使用预先下载的 `PP-OCRv5_server_det` / `PP-OCRv5_server_rec`。它改变原 runner 的 `min/736` 检测缩放配置，不等同上游默认配置。两种尺寸的观察不是识别质量或普遍性能增益评测。参数及模型选择依据为 [PaddleOCR v3.0.0 官方教程](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.0.0/docs/version3.x/pipeline_usage/OCR.md)。

PaddleOCR / PaddleX / PaddlePaddle 均保持 **3.0.0**，NumPy 为 1.24.4；没有下载替换依赖、更新 SDK 或使用 `KMP_DUPLICATE_LIB_OK`。该隔离环境继承本机系统库，并非可移植干净镜像。目标验收没有传入旧 `drop_score` 参数；报告保留的旧配置加载器默认值 0.5 不是 3.0.0 实际过滤参数，实际阈值 0.0 单独记录。未证明阈值配置迁移、PPStructure 兼容、识别准确率或发布批准。

```powershell
# 固定原创应用，有界监督；记录存在时换一个新 name，不覆盖历史。
.local-demo-temp/ocr-v3/Scripts/python.exe evaluation/business_quality_v1/v3_supervise.py --name v3-target-rerun --timeout 180 --application-acceptance
```

四条真实 2.9.1 记录已在当前冻结源码重跑：`real-v2-resolution-{default,old,new,override}.json` 均通过，监督耗时分别为 13.13、13.66、13.99、13.10 秒。八项固定分数夹具重新绑定当前源码，三项故意失败仍为 `FAILED`，全部符合预期。自动化检查为原创应用 53 项通过、`tests/test_upgrade_review_depth.py` 25 项通过、新评测脚本编译检查通过。

冻结应用清单 SHA-256：`93f1e5b22512bfe2e965a400c1107ab0d2586a1a0152b92b3495d2d06a8ed1cb`；`pipeline.py`：`cb5c6bcd9d81cf53fd94e2b703e5448cdb9889bce592f89d7363453a9922703b`；未修改的消费者：`769e965d3f9a318ecfa8bf3e0b015e43154d7e345059cfb0734e34cd78cb91e1`。完整逐文件身份及报告哈希在结构化汇总中。
