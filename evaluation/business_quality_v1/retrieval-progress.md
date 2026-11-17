# 检索执行记录（2026-11-14）

T3 当前进度：冻结运行器、固定题集、晋级门槛及来源验证已实现；等待主线服务源码冻结信号后重新生成最终报告。

- 测试先行：初始13项测试观察到明确失败；随后逐次观察到 warm 证据漂移、通用资料模块分类、依赖环境漂移、来源链接冒充、first-pass 慢但 warm 快的错误晋级测试失败。实现后19项测试通过，日志在 `.local-demo-temp/business-quality-20261114/retrieval-{red,red2,red3,red4,red5,red6,green}.log`。成本门槛同时约束 first-pass 与 warm P95，而不是只按命中精确缓存的耗时建议启用。
- 固定22题/26事实。11道旧开发回归、11道开发者口语/负例验收题；不是盲评，未声称业务准确率。
- 初次 BM25 已实际查询当前1794块语料。当前主线仍在修改，初次运行不作为最终可发布成绩，最终身份需在冻结后重跑。
- 完整模型文件存在：本机缓存 BGE-small-zh-v1.5，模型权重95,827,648字节；仓库本地 BGE-reranker-base，权重1,112,206,140字节。
- 首次加载失败：`.venv-maintenance` 当时缺少 `torch`，原始原因 `No module named 'torch'`；同时缺向量检索可选库。历史探测原文保存在 `.local-demo-temp/business-quality-20261114/retrieval-model-probe.log`，不能把首次失败写成当时已运行成功。
- 为完成真实对照，已在同一维护环境按仓库 `requirements-quality.txt` 安装 CPU torch==2.4.0+cpu、sentence-transformers==5.3.0、transformers==5.4.0，保持 numpy==1.26.4。安装前后清单显示原有包0变更，新增19项可选依赖；`pip check` 为 `No broken requirements found.`。没有下载模型、没有复用旧环境；CPU wheel 为200.2MB缓存文件。安装命令和报告见本地 `retrieval-{torch,quality}-install.{log,json}`，清单为 `retrieval-pip-{before,after}.txt`。
- 安装后 BGE-small 的真实本地编码已成功，2条文本得到 `[2,512]` 有限向量，cold加载加编码44.2秒，模型内容身份 `ba70b41bd018d3b7d57bf31b7111833db90564128f699220113ae16d00c155e5`。原始日志 `retrieval-embedding-probe.log`。
- 安装后的重排探测曾触发已有 `asset_identity` 用 `read_bytes()` 一次读1.1GB文件的 `MemoryError`，错误日志 `retrieval-model-probe-installed.log` 保留。主线现已改为流式读取并保持相同摘要；评测自身冻结哈希同样使用分块读取。
- 真实 `contextual_rrf` 初次运行已开始构建完整文档窗口向量。其临时身份为 `a3ccd48636a98d1b80cdcd6069f2b3bb6ca45878967b4f0043e2197ce53a9a2c`；运行期间 `joint_knowledge.py` 变化，下一次 BM25 启动已正确拒绝这个过期身份。首次向量构建完成后只复用按模型/完整原文键控的真实向量缓存，旧逐题成绩不能当最终成绩。
- 首次真实向量构建已完成：首题698,995.1ms，生成3484×512的float32文档向量缓存（7,135,360字节）。这是历史临时运行的建库成本，不能冒充最终已预热运行的冷请求时间。
- 随后曾用身份 `124cc96b499bb9391dc93ebf5417cc56c433f19d8ec2995803bbf9df4bed499a` 跑完 BM25 与真实RRF；BGE真实重排完成前5题，每题未缓存41.9–50.9秒。主线据真实否定模块范围缺陷修复 `query_module` 后该身份已过期，重排进程按主线指示停止。旧报告仅为失效诊断记录，不作为最终当前效果；等待最终确认后新目录重新冻结运行。重排精确缓存原本只在进程内，已停止进程的原始pair logits没有导出，不能声称已复用这5题的模型分数。
- `missing-internal-approval` 的直接检索级范围表现另行披露：运行器绕过普通查询入口的内部事实业务保护；普通入口会先拦截这个问题。不能把组件级候选混杂解释成端到端产品已经生成错误审批结论。标签没有改写，生成拒答仍为 `NOT_RUN`。
- 更新可选环境后曾尝试一次全后端测试，但收集阶段导入 `faiss/swigfaiss.py` 时 Windows 原生异常 `0xc0000008`，没有得到全套通过结论，详见 `retrieval-service-suite.log`。主线负责最终全量验证，本分支不继续并发启动大套件。
- 生成评测 `NOT_RUN`，外部模型调用0；只比较检索/组装上下文/完整来源 oracle。两个内部事实问题未执行生成拒答测试。

禁止把可用的模型文件等同于已完成神经检索运行；最终神经请求若仍缺依赖，状态必须为 `NOT_AVAILABLE`，并保留回退原因，不能把 BM25 回退记为 hybrid 成功。
