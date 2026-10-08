# 面试封版验收

四道查询从当前 UI 的 `interview_queries` 读取，三种审查输入从 `DEMO_CASES` / `demo_application` 读取。每组重复两轮，保留完整响应、提交文件、源码指纹和工作区信息。生成使用当前配置的 DeepSeek；用户明确授权仅公开资料和学生原创演示数据向该 API 传输。

`local-acceptance.json` 为修复前首次记录，发现仅识别问题混入 `rec=False` 用法；不能算正确成功。`local-acceptance-final.json` 是修复参数条件约束和升级证据角色占位后的重新运行，没有覆盖第一次记录。

三个 Agent 分支两轮分别为 `supported_risk` / `needs_verification` / `needs_verification`，均 `runtime_verified=false`。这说明演示分支结论可重复，不证明所有变更影响都被发现。

四个 RAG 请求两轮均返回 `OK`，但多个复合问题保留 `PARTIAL_SUPPORTED` 和缺口。第一题修复后不会再把关闭识别当成仅识别的等价用法；两版迁移保留调用返回列表及结果字段证据。内部兼容性题无法据官方資料证明下游应用已兼容。不能把 8/8 请求返回写成 100% 回答准确率。

前两份记录与本地 CPU 策略比较存在并行资源竞争，所记录时间只用于定位与复现，不作为公网 SLA 或 P95。

`deployment-profile-acceptance.json` 使用最终选型记录和未安装 BGE 资产的公网等效配置重新运行：8 次 RAG 请求均返回 OK，耗时 20.09–53.98 秒；6 次 Agent 的三个分支状态各重复一致，耗时 2.35–42.55 秒，全部 runtime_verified=false。RAG 六次标记 PARTIAL_SUPPORTED，两次 NOT_ASSESSED；后者不是完整正确的证明。没有请求异常记录，但模型输出仍有波动，不能称八题全部正确。

记录中的模型核验不是专家标注。此次封版停止新增业务，复合回答完整性和独立人工标注作为后续改进项。

运行：`python evaluation/interview_release_20261009/run_acceptance.py --output 新报告路径 --repeats 2`。默认调用已配置的外部生成服务，运行前须确认数据范围与授权。已有报告不会被覆盖。
