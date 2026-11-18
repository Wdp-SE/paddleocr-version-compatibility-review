# 当前 PaddleOCR 代理语料检索对照

这是内部研发知识与同一应用变更审查 POC 的组件证据检索实验。真实发布者保持 PaddlePaddle/PaddleOCR；不把官方资料当作原创应用的输出契约、企业内部事实或升级通过证明。PPStructure 题只验证相关接口资料检索，不代表运行过该分支。

## 当前发布验收

从仓库根目录使用新输出路径执行下列命令。该入口走实际默认检索路径，检查 22 道固定题和 9 道开发改写题的必要事实、版本与模块，记录语料与实现指纹。模型生成和人工回答质量复核另行验证。

```powershell
.\.venv-maintenance\Scripts\python.exe evaluation/business_quality_v1/production_retrieval_acceptance.py --output .local-demo-temp/retrieval-verification
```

## 检索策略开发实验

`cases_retrieval.json` 固定 22 题、26 项必要来源事实：11 道已见开发回归题，11 道开发者新写口语题与负例。新措辞属于冻结验收集，但来源任务已知；这些都不是独立业务专家盲评。事实标注包含版本、模块、允许路径、必要字面片段及完整来源 oracle。`version-conflict` 要求澄清；两个缺内部事实问题不设置虚构答案，检索相关资料不能当作生成拒答已经通过。

允许路径是预先固定的来源见证，不是所有语义等价资料的穷尽清单。其他正确说明可能没有被这一严格字面规则计分；因此低覆盖可定位候选缺口，但不能直接断言相同比例的回答错误。通用资料标为 `general` 时单列未分类数，不把它一律冒称为已确认的其他模块。

运行器直接调用 `configured_evidence_search`，用于隔离检索策略，未经过普通公开查询入口的业务路由/缺内部资料保护。特别是 `missing-internal-approval`，普通查询入口会先拦截企业内部运行/审批事实问题；这里的检索级跨模块候选不应误称为产品向用户生成了跨模块回答或批准上线。这个控制题保持原标签，单独披露路由限制；没有运行端到端生成拒答，状态仍是 `NOT_RUN`。

策略固定为 `contextual_bm25`、`contextual_rrf`、`contextual_rrf_rerank`，共用现有 `configured_evidence_search`。Top-K=5、候选窗口=40、每源最多4窗口、字符窗口=384；不修改语料、服务策略或发布配置。真实 dense 候选与真实 rerank 调用成功才算相应策略运行；模型回退即 `NOT_AVAILABLE`，即使回退文本命中也不能计作神经策略成功。

冻结身份覆盖服务全部 Python 源码、完整当前语料及上游来源文件、模型权重与全部 tokenizer 文件、问题与评分程序、配置预算和关键库版本。开始和结束均复核；保存结果再次比较时从当前原始语料重建跨度并重算最终覆盖，拒绝伪造文本、错身份和过期报告。

每题分别记录候选池、最终检索跨度、生成前组装上下文、完整来源 oracle 的字面事实覆盖。oracle 用于区分语料有资料但没有检索到，以及截窗/预算丢失；没有调用生成 API，不报告回答正确率、语义支持或生成拒答准确率。原始 first-pass、两次 warm 延迟与模型回退都保留。每个策略使用独立进程；首题可包含全量向量构建，其余题的 first-pass 不称为进程冷启动。文档向量复用现有适配器按完整文本和模型身份键控的磁盘缓存，逐运行记录缓存文件哈希；查询仍由真实模型编码。warm 还复用重排器的进程内精确缓存，不能当作未缓存线上神经推理成本。

晋级仅形成建议：逐题已有必要事实不丢失、错版本/错模块不增加、负例范围不出错、验收题至少新增一个事实，first-pass 与 warm P95 均 ≤ 5000ms 且 ≤ 各自基线10倍，防止 warm 精确缓存掩盖未缓存重排推理成本。首次建库/初始化单独记录；22题 nearest-rank P95不会由唯一首题建库时间单独决定。回退、缺题、身份变化、warm 证据漂移均不晋级；开发集的收益不能替代验收收益。即使门槛通过，也不直接写发布选型配置。

从仓库根目录运行，每条 run 命令必须是独立进程；默认输出目录 `.local-demo-temp/business-quality-20261114/retrieval-final`，文件已存在时拒绝覆盖。源码变化后使用新的输出目录重新冻结，不能沿用旧成绩。

```powershell
.venv-maintenance/Scripts/python.exe -m pytest -q -p no:cacheprovider versioned-rag-service/tests/test_business_retrieval_benchmark.py
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/retrieval_benchmark.py freeze
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/retrieval_benchmark.py run --strategy contextual_bm25
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/retrieval_benchmark.py run --strategy contextual_rrf
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/retrieval_benchmark.py run --strategy contextual_rrf_rerank
.venv-maintenance/Scripts/python.exe evaluation/business_quality_v1/retrieval_benchmark.py compare
```

开发调试可用 `run --split dev`，最终比较读取 `--split all` 报告，分别显示 dev 和 acceptance。本轮没有按 acceptance 成绩调节服务检索代码或更换题目。指定 `PADDLEOCR_EMBEDDING_PATH` / `PADDLEOCR_RERANKER_PATH` 可使用其他完整本地模型，但必须重新冻结和比较。默认仅寻找本机缓存 BGE-small-zh-v1.5 与仓库本地 BGE-reranker-base；离线加载，不触发模型下载。

实际测量、阻断原因与测试记录见 `retrieval-progress.md`；最终 JSON/Markdown 由 `compare` 命令生成。
