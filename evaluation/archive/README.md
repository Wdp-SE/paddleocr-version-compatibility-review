# 历史评测归档说明

本目录页说明仓库中仍保留的旧评测目录及其边界。旧领域的题集、语料指纹和报告不属于当前 PaddleOCR 产品，不能作为当前成绩。少量历史文件仍供回归测试检查过期成绩拒绝机制，不能仅凭目录年代删除。

当前评测入口：

- [`../retrieval_upgrade_v2`](../retrieval_upgrade_v2/README.md)：当前结构检索、真实 BGE 混合与重排对照；发布指纹须通过验证。
- [`../retrieval_upgrade_v2/task_verification.md`](../retrieval_upgrade_v2/task_verification.md)：真实生成和审查复核，保留未完整成功的案例。
- [`../internal_workflow_v1`](../internal_workflow_v1/README.md)：前轮内部工作流验证；是否适用于当前版本须核对指纹。

Autoware、DolphinScheduler、Seeed/Jetson、PP-Human 等目录属于历史材料。PaddleOCR 的较早评测保留原始题目、失败项与指标，以追溯技术选型；没有通过当前发布指纹验证的报告，不得作为当前成绩。

更换语料、代码或评测题集后，只有与当前发布指纹一致的冻结报告才能被工作台展示。旧结果不得迁移为新语料的基准，也不能用来声称新领域有效。
