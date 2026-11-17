# 版本证据与工作流深化回归

`cases.json` 是12道开发回归题，包含等价迁移问法、方向分类条件、参数交互、六项组合要求、字段对应缺口和内部事实边界。题集已参与调试，不是独立盲测。预期事实不发送给生成模型。

从根目录执行（每次必须使用新的输出目录）：

```powershell
python evaluation/maintenance_acceptance_v1/run.py --url http://127.0.0.1:8774 --suite depth12 --generate --output .local-demo-temp/depth-new
```

`curated_context_probe.py` 对预先声明的两题比较正常检索与开发者选定的完整原文，使用同一真实生成器和核验器，不注入参考答案。来源控制有范围与字符预算；它用于判断断点是否位于上下文选择，不能充当独立专家金标。需按本地后端启动配置设置语料与模型环境后执行。

`low_score_runtime_probe.py` 在已有 PaddleOCR 2.9.1 环境中运行原创应用与原创合成图片，记录阈值0.5/0.8的真实输出、置信度和输入哈希。该实验通过运行参数覆盖阈值，明确 `configuration_loading_verified=false`，不能证明提交的JSON/YAML配置已在目标环境加载，也不能据输出变化认定业务质量提高。

所有报告区分模型完整性、范围说明、静态发现、夹具记录和实际OCR。独立人工正确性评审仍待完成；代码、语料或策略变化后需重新核验。
