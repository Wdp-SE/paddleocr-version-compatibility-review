# 应用维护验收

这一组验收对应当前业务范围，避免把检索覆盖、生成完整性、静态发现和运行回归混成一个分数。

- `cases.py`：十五个页面示例的事实审阅要点，以及四个独立措辞探测。事实不传给检索器或生成器。措辞探测不是独立业务专家盲测。
- `run.py`：按当前服务身份保存实际检索/生成结果、来源、缺口、耗时及补查轨迹。默认只检索；`--generate` 会使用后端配置的模型。可用 `--only example-11` 复测特定失败项。
- `review_workflows.py`：在指定输出目录新建原创应用的本地 Git 仓库，生成真实基准和候选提交，验证旧结果消费、字段适配、调用层缺失三种路径；检查回归记录跨提交失效，再运行配置审查。不能复用已存在的输出目录。
- 配置验收也生成真实基准、候选及开发者调整后的提交；确认同版本参数变化的证据、复查身份与旧回归记录失效。生成了提交不代表执行过 OCR。
- `parent_context_probe.py`：受控比较同一批已检索来源的短窗口与有界完整原文块，同时使用真实生成器及核验器。只改变上下文，不加入参考答案、不增加来源；它不是人工标注的 oracle 或独立盲测。
- `maintenance30.json`：三十个扩展维护问题及事实审阅要点，覆盖调用条件、批量输入、参数迁移、结果类型和内部事实边界。由开发者与 AI 辅助设计，已用于定位失败，属于回归集，不是独立盲测。
- `acceptance_report.py`：分开报告请求失败、无证据、版本澄清、模型完整性与版本范围异常；`human_accuracy` 保持空值，独立人工验收仍待完成。
- `--suite depth12`：运行版本语境、组合要求及内部事实边界的12道开发回归题；控制实验与真实阈值对照见 `evaluation/depth_quality_v1/README.md`。范围说明单独统计，不计作事实题完整回答。工作流验收也包含JSON配置声明链，仍不证明文件运行时加载。

从仓库根运行，输出目录建议放在忽略的 `.local-demo-temp/`：

```powershell
python evaluation/maintenance_acceptance_v1/run.py --output .local-demo-temp/maintenance-check
python evaluation/maintenance_acceptance_v1/run.py --generate --output .local-demo-temp/maintenance-generated
python evaluation/maintenance_acceptance_v1/run.py --suite all --generate --output .local-demo-temp/maintenance-all-new-run
python evaluation/maintenance_acceptance_v1/review_workflows.py --output .local-demo-temp/workflow-check
python evaluation/maintenance_acceptance_v1/parent_context_probe.py --output .local-demo-temp/context-probe-new
```

模型判定为完整不代表人工确认正确；十四题返回内容不等于十四题全部正确。来源真实也不能证明引用足以支持主张。配置对照、静态报告和夹具不能证明真实 OCR 已运行或企业 ROI 已验证。代码或资料变化后需重新验收，原始结果按服务/语料身份保留，不覆盖成新成绩。

每次运行必须使用新的输出目录，已存在的目录会被拒绝，`--only` 也不会覆盖完整验收记录。结果保留问题集哈希、运行器哈希及后端的语料、策略、服务和审查代码指纹；逐题的预期事实不发送给模型。每次响应的 `X-Service-Identity` 必须与运行前后工作空间指纹一致；运行中更换服务标为 `CHANGED`，旧服务未提供指纹或结束检查失败标为 `UNVERIFIED`，两者均不能形成有效的一致版本验收。`run_valid` 仅表示题数齐全、服务身份一致，不代表回答正确或所有请求成功；仍须审阅 `summary` 中的失败、空召回与人工待审状态。

首轮检查结果与已知不足见 [2026-10-19 质量记录](closeout-20261019.md)，后续复核见 [同日复核记录](audit-20261019.md)。发布冒烟门槛还会拒绝 `unknown` 提交，以及源码、语料、审查或检索配置指纹不匹配的版本；HTTP 200 不代表前端构建已核对。
