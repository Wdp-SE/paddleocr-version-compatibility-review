# 原创应用配置与升级运行验收

最终状态以 `runtime-summary-final.json` 和 `docs/maintenance/2026-11-14-business-quality-acceptance.md` 为准。下文保留中间尝试记录。后续集成已将默认 `process(path)` 与显式 `process_configured(...)` 分离，重新执行并绑定了 `real-v2-verified-{default,old,new,override}.json` 四条当前源码记录；旧 `runtime-summary.json` 已标为历史。历史3.0.0原生退出发生于分离前，不能作为当前源码的兼容结论。

范围：公司内部 OCR 封装维护的学生 POC；只执行 `examples/paddleocr_document_app` 的作者原创应用。不执行提交的应用，不扩展为任意仓库兼容验证，不验证 PPStructure。官方依赖版本与应用 Git 基线/工作区文件哈希分别记录。

运行命令：

```powershell
.venv-maintenance/Scripts/python.exe -m pytest examples/paddleocr_document_app -q -p no:cacheprovider
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/runtime_acceptance.py --output .local-demo-temp/business-quality-20261114/runtime-fixtures.json
.venv-maintenance/Scripts/python.exe examples/paddleocr_document_app/run_upgrade_demo.py --create-fixtures --fixtures .local-demo-temp/business-quality-20261114/ocr-fixtures
# 在现有隔离 OCR 环境运行，runner 拒绝缺失的本地模型；不下载模型。
.local-demo-temp/ocr-v2/Scripts/python.exe examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/business-quality-20261114/ocr-fixtures --config .local-demo-temp/business-quality-20261114/runtime-configs/old.json --output .local-demo-temp/business-quality-20261114/real-v2-json-old.json
.local-demo-temp/ocr-v2/Scripts/python.exe examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/business-quality-20261114/ocr-fixtures --config .local-demo-temp/business-quality-20261114/runtime-configs/new.json --output .local-demo-temp/business-quality-20261114/real-v2-json-new.json
.local-demo-temp/ocr-v3/Scripts/python.exe examples/paddleocr_document_app/run_upgrade_demo.py --fixtures .local-demo-temp/business-quality-20261114/ocr-fixtures --output .local-demo-temp/business-quality-20261114/real-v3-default.json
```

夹具矩阵的 `expectations_met` 表示观察到了预先声明的成功和失败，不能改称所有案例运行成功。失败案例的 `runtime_status` 保持 `FAILED`。`file_changed_not_adopted` 中新 JSON 被显式旧阈值覆盖，因此未满足新阈值输出要求。`v3_default_entry_fails` 的独立适配器探测通过，但旧应用入口仍失败。`v3_candidate_absorbs_shape` 仅用固定夹具注入预定目标适配器，消费者文件哈希与其他案例相同；它不是已发布候选或真实 OCR 兼容结论。`unchanged_call_runtime_bad_type` 验证调用表面不变时仍可能运行失败；该 runner 不替代静态审查报告。

实际 OCR 采用两张原创中文合成扫描页与一张空页。记录包含样本字节哈希、应用文件哈希、Git 基线及脏工作区标志、依赖版本、配置字节哈希及来源、构造参数、归一化和应用入口各自结果。固定分数夹具验证过滤边界；实际 OCR 只观察这三张样本。不能由它们计算业务识别准确率或推断真实企业收益。

原始 RED/GREEN 日志、夹具及实际 OCR 报告保存在 `.local-demo-temp/business-quality-20261114/`；逐项状态见其中 `config-progress.md`。本目录 `runtime-summary.json` 汇总当前六份实际 OCR/受控进程记录的状态和字节哈希、50 项程序测试、八项夹具预期及限制，保留真实 3.0.0 不可用状态。结果绑定记录中的源文件哈希，后续源文件或样本变化须重跑。

## 2026-11-14 当前执行记录

统一轻量环境运行整个原创应用测试目录：**50 passed**，原始输出 `final-app-tests.log`。本轮先观察配置入口缺失、运行记录缺失、置信度对应遗漏、大整数校验溢出、执行中源文件变化及原生进程退出缺记录等 RED，再实现并复测。测试数量是程序回归结果，不能解释为问答正确率或 OCR 识别准确率。

| 实际 OCR 2.9.1 路径 | 配置有效值及来源 | 当前结果 |
| --- | --- | --- |
| 无 JSON 默认值 | 0.5 / default | PASSED |
| old.json | 0.5 / json | PASSED |
| new.json | 0.8 / json | PASSED |
| new.json + 显式 0.5 | 0.5 / explicit_override | PASSED |

四份 `real-v2-*.json` 记录均通过实际应用入口、文字与分数对应、字段类型、页序及空页检查；执行前后源文件身份相同，且和当前源码逐文件哈希一致。四份记录使用相同三页样本、相同未修改消费者（SHA-256 `769e965d3f9a318ecfa8bf3e0b015e43154d7e345059cfb0734e34cd78cb91e1`）。实际依赖是 PaddleOCR 2.9.1、PaddlePaddle 2.6.2、NumPy 1.26.4；复用现有模型，没有下载。

这三张清晰样本在 0.5 与 0.8 下的 JSON 文本相同，**没有观察到真实 OCR 过滤输出差异**。有效值与配置来源记录证明实际工厂采用了不同阈值；“旧值未采用新配置会违反预期输出”的负例及阈值两侧/等于边界来自独立固定分数夹具。不能将二者混为真实业务收益。

固定分数矩阵的八个案例均符合预期；其中三项仍明确保持 `FAILED`：`file_changed_not_adopted`、`v3_default_entry_fails`、`unchanged_call_runtime_bad_type`。目标形状候选只在夹具中成功，不能称真实 OCR 3.0.0 升级通过。

OCR 3.0.0 使用已有 Paddle/PaddleX 3.0.0、NumPy 1.24.4 及本地 PP-OCRv5 server 模型。先前普通进程退出但没有产生 Python 报告，只能视为无结论的环境尝试；另一个普通尝试已停止以避免并发模型资源竞争。最后仅保留一个有超时和外部失败记录保护的受控子进程：运行 492.73 秒后，原生子进程以 **3221225477** 退出，未产生子进程 JSON 或 Python traceback。原始日志止于本地检测/识别模型初始化。最终记录 `supervised-real-v3-default.json` 状态为 **NOT_AVAILABLE**；当前真实 3.0.0 入口和升级兼容状态仍为未验证，不能以夹具或历史适配器报告补为通过。

可通过统一轻量解释器启动外部保护器；子进程仍只执行固定原创应用及固定本地 OCR 环境：

```powershell
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/runtime_acceptance.py --real-runtime v3 --output .local-demo-temp/business-quality-20261114/supervised-real-v3-default.json
```

保护器为原生退出、无报告或超时写入 `NOT_AVAILABLE`，并保留退出码、原始日志和应用身份；这种状态不能推断为升级风险或兼容成功。普通 Python 异常及应用契约失败由子进程写入 `FAILED`。执行中应用源码改变时记录失效为 `INVALIDATED`。

## 静态配置演示与实际运行入口的区别

全量集成回归发现旧演示生成器依赖替换真实工厂中的字面量调用。真实工厂新增配置校验和条件分支后，生成结果仍包含动态分支，受限静态分析正确保留缺口，没有形成可确认的阈值变化。

现已修复 `demo-ui/services/configuration_workflow.py` 的演示输入生成：直接数值与 JSON 两种模式分别生成明确的原创静态调用层夹具，保留其他应用文件；JSON 使用与真实加载入口一致的平面 `schema_version/drop_score` 结构。生成调用层的文档字符串包含 `developer_original_static_fixture`、原始调用层 SHA-256 和 `runtime_verified=false`，明确说明经过简化且未执行。静态分析规则没有放宽，真实应用代码和已有运行记录身份未变。

重现原 Agent 失败：`config-demo-agent-red.log`，1 failed；修复后完整 Agent 配置/API 文件：`config-demo-agent-green.log`，4 passed（1 条既有 LangGraph 弃用提示）。UI 配置测试先 RED 3 failed / 2 passed，后 GREEN 5 passed；同时验证直接/JSON 两种输入均产生 0.5→0.8 的明确声明变化、没有静态缺口、没有运行验证声明，并保留原始应用文件。静态演示可操作不代表真实 JSON 加载或 OCR 运行通过，仍分别读取上面的运行记录。
