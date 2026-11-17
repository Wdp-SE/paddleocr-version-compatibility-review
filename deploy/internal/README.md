# 内部服务部署与验收

这套配置将内部入口与公网演示隔离。业务仍是登记的文档处理应用及 PaddleOCR v2.9.1 → v3.0.0；不是任意仓库扫描平台。代码实现了认证边界、持久任务、独立 Worker、CI 来源验证与人工记录；提交本配置不代表企业生产环境已经验收。

## 部署前必须提供

1. 企业身份服务：可信 HTTPS issuer、API audience、JWKS。仅接受 RS256；使用访问令牌，不能把其他 audience 的登录 ID token 当作 API 凭证。前置认证代理负责登录、清除客户端伪造身份头，再向前端转发 Authorization。前端不允许绕过代理访问；API 对每次请求独立验签。
2. 服务身份刷新代理：定期将短期 Worker 访问令牌写入私有目录 `access-token`，采用原子替换。挂载整个目录而不是单个文件，Worker 每个 HTTP 请求重新读取；身份代理需要在过期前完成刷新，失败时不回退匿名。需要部署方提供 IdP 的刷新集成。
3. PostgreSQL、受限运行账户、备份目标与 TLS。示例中的数据库 owner 仅用于初始化；运行 URL 必须使用单独账户，不使用超级用户。生产数据库 URL 不进入 Git、构建参数或普通日志。
4. 只读应用 Git 仓库与服务端登记表。按 `config/internal_applications.json` 的结构创建私有 `applications.json`，repository 指向 `/repos/document-ocr`。只由可信同步流程更新 Git 对象；服务不接受客户端仓库 URL，不执行 checkout/hooks 或待审代码。
5. 已验证的语料发布目录、批准的镜像 digest、API TLS 证书与 CA。API/Worker 使用同一镜像，包含 RAG、Agent 和编排代码；不允许任意热更新 Worker 源码。

## 初始化与启动顺序

从仓库根目录构建，指定经过审核的 Python 基础镜像 digest：

```bash
docker build -f deploy/internal/Dockerfile --build-arg PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:<approved-digest> -t <private-image>:<commit> .
```

`requirements.lock` 固定 CPython 3.11、Linux amd64 的直接及传递依赖和轮子 SHA-256，安装使用 `--require-hashes`。本次只验证了下载文件哈希与本地模块测试；容器启动和漏洞扫描仍需在部署环境验收。更新依赖时重新解析、锁定并跑回归，不能手改哈希。基础镜像与系统 apt 包还需要组织镜像仓库和构建扫描，不能仅凭 Python 锁文件宣称整个构建完全可重现。

将 `internal.env.example` 复制到部署机器的 `internal.env`，填真实 IdP 配置；私有文件不提交。Compose 的变量使用 `:?` 强制提供镜像、配置、Git 仓库、身份目录、数据库 URL 与 TLS 密钥位置。默认不向宿主机发布端口；将企业认证代理接入 review 网络，内部 TLS 证书 SAN 必须包含 `api`。

先启动数据库，再用迁移账户在离线管理进程执行：

```bash
cd versioned-rag-service
python -m src.internal_operations migrate
```

当前 schema 为 2；管理迁移会保留 schema 1 任务并增加 review_graph_state 检查点表，初始化幂等；运行模式只检查 schema，不自动修改结构。迁移期间不接收新任务。迁移账户可建表；运行账户只授予当前 schema 的 USAGE 和表 SELECT/INSERT/UPDATE（应用不执行删除）；下一版 schema 需提供明确迁移与回滚方案。

配置运行账户和文件权限后再启动 API、Worker、前端。`docker compose -f deploy/internal/compose.yaml config` 应通过，随后在部署网络中验收，不应直接暴露未配置代理的前端。

## 数据出口与授权

`access-policy.example.json` 的 principals 为空，默认没人有权使用。使用真实 JWT sub 配置应用动作：read、submit、review、worker、admin。普通发起人不能冒充审核人，非所有者需要 review 才能读任务。访问策略每次访问重新读取；错误配置拒绝访问。

`external_models` 默认为空，生成也默认禁用。需要生成时，须同时批准应用资料等级、实际服务商、出站网络域名、服务商数据保留政策及密钥管理。当前代码检查最终拼接提示词的大小和常见密钥格式；这是补充检查，不是完整 DLP。真正内部敏感数据需要经过组织评估。不能把外部模型账户和文档里的工具指令当成授权。

## 应用知识查证

`POST /internal/application-knowledge` 接收 `application_id`、Git 引用 `head`、组件资料版本 `version`、问题 `query` 及可选的 `generate`。默认仅检索证据。服务先校验该应用的 `read` 权限，从已登记仓库解析提交并读取允许的文件范围。返回解析后的 `application_commit`、应用证据及快照缺口。客户端不上传资料或指定仓库路径。

`version=auto` 使用可选的 `entrypoint`，从登记表的 `模块路径::函数名 → 依赖文件路径` 映射中查找选项。依赖文件需登记在 `roles.dependency`。服务从同一提交读取函数定义和唯一的 PaddleOCR 精确版本声明，返回 `entrypoint_context` 及两处来源行号。缺少入口、缺少文件或依赖声明不明确时，返回 `NEEDS_ENTRYPOINT_CONTEXT`。可显式选版或用 `version=all` 查看两版资料。入口与依赖关系属于登记和提交信息，实际运行使用的依赖仍由运行记录确认。

需要回答时设置 `generate=true` 并满足应用模型外发与预算策略。组件事实和应用事实带各自的引用。应用契约缺失时，系统可以给出有依据的组件事实和资料缺口。测试定义属于应用资料，实际回归通过须有绑定相应提交的运行记录。

## 任务与故障处理

提交身份包含应用/基线提交、登记策略、资料、检索策略、实际分析及编排代码、访问策略。相同身份重复提交返回同一任务；策略变更需重新发起。持久任务由 Worker 领取，60 秒租约、10 秒心跳；过期可接管，总计最多三次执行。旧 Worker、取消任务不能覆盖结果。模型在途请求不能保证立即取消或恰好计费一次。

显式重试与新提交共同遵守每主体 8、全局 64 个活动任务上限。生成默认最多 4 个并发租约，可按应用配置 UTC 自然日请求预算；这是请求预算，不是精确 token/货币结算。当前 Worker 将两条审查图的安全节点检查点保存到同一数据库，并核对请求、构建和租约后恢复；模型相关节点在途状态不确定时以 GRAPH_REPLAY_UNSAFE 停止自动重放。没有声明跨 Worker 恰好一次模型调用或计费。普通 RAG HTTP 查询仍是请求内图，不具有跨请求恢复。

`/health/live` 只检查进程；`/health/ready` 检查数据库 schema、真实基础索引就绪以及完整构建身份。可选模型不是基础就绪条件。管理权限可读取 `/internal/metrics` 的状态数、有效 Worker 租约与最久排队时间。空闲 Worker 没有独立常驻健康注册；需结合容器/进程健康监控。请求日志记录 request_id、路由、状态、耗时，不记录源代码、JWT、问题或完整异常。

对队列积压、租约接管、持续失败、模型降级、DB 连接失败和备份失败配置告警。P95、容量、恢复时间和费用目标需压测/演练实测，不能沿用演示数字。部署时使用出口代理/防火墙限制网络；Compose 网络本身不构成外部模型域名白名单。

## CI 与人工记录

任务 CI 入口使用独立 GitHub OIDC issuer/audience。应用策略 ci 必须明确 repository、workflow_ref、40 位 workflow_sha，且验证令牌的 `job_workflow_ref/job_workflow_sha`，因此应使用固定 SHA 的**受信可复用工作流**。普通员工 JWT 不可伪造 CI 记录。当前仓库的边界测试工作流只运行测试，不自动向生产审查系统提交兼容声明。

受信 CI 的 POST `/internal/tasks/{id}/ci` 请求应包含 `application_commit`、当前 `report_fingerprint`、`run_id`、PASSED/FAILED、完整 `input_sha256` 及 `environment`。环境必须有 paddleocr、paddlepaddle、python、runner_os、model_sha256、container_digest。提交、测试和契约哈希必须匹配报告快照。受信工作流自行运行真实回归并产生声明，不能给未受信应用代码同一个有 OIDC 权限的执行环境；需要隔离测试执行与证明提交，并限制可复用工作流的变更权限。

`GET /internal/tasks/{id}/report` 返回 `regression_coverage`，供运行器按报告中的 `check_id` 对应测试项。受信 CI 可以随现有载荷提交 `check_results`，每项包括 `check_id`、`status`（PASSED、FAILED、SKIPPED）、`execution_kind`（REAL_OCR、FIXTURE）和 `artifact_sha256`（运行产物的 SHA-256）。产物由受信运行环境留存，服务记录其摘要及运行编号。只展示同一报告、同一应用提交下每个工作流的最新已接收尝试。夹具通过单独展示，历史整单记录及漏报项保留逐项未运行状态。

人工记录支持 ACKNOWLEDGED、DEFERRED、CONTINUE_REVIEW。继续评审需要不同审核人、当前目标版通过的受信 CI、风险逐项理由和限期。CI 验证令牌中的调用工作流身份、run_number、run_attempt；每个调用工作流最新已接收的序号/重试不能失败，较早成功不覆盖较新失败。旧记录缺少序号身份时需要新 CI，不沿用为放行依据。即使全部满足，返回的 `upgrade_approved` 仍为 false：组织既有发布流程才有放行权限。系统只知道已接收记录，不宣称实时掌握 GitHub 全部运行；CI 来源可信也不证明测试覆盖充分。

## 语料发布与回滚

`src.internal_operations release` 先加载并验证来源哈希、固定版本、索引、图片证据及检索配置，再复制到内容寻址目录。管理员提供评测 gate，包含 passed=true、tree_sha256、service_code_fingerprint、evaluation_sha256；评测结果必须在可信离线流程实际产生，不能手填通过来替代评测。

```bash
python -m src.internal_operations release <candidate-corpus> <release-directory> <gate.json> --activate
# 已有活动版本时必须显式提供读到的 release id
python -m src.internal_operations release <candidate-corpus> <release-directory> <gate.json> --activate --expected <current-release-id>
```

切换使用排他锁和比较当前身份，原子更新 active.json。服务启动再核对全部文件，检测篡改；Worker 不改变活动目录。运行中的服务保持已加载版本，切换后滚动重启 API/Worker。回滚时激活保留的旧目录/gate，并部署其匹配的服务代码；不能混用旧证据、新代码的旧成绩。限制目录写权限到发布账户，保留旧发布物；工具不自动删除历史版本。断电持久性、共享文件系统原子语义需在目标存储验证。

## 备份与恢复演练

PostgreSQL 使用对应版本 `pg_dump -Fc`，通过私有 PGPASSFILE/平台服务身份连接；不要把带密码 URL 拼进命令或日志。备份包含任务输入、结果、CI 与审计，必须加密存储并限制访问。恢复使用 `pg_restore --exit-on-error --no-owner` 到**新建的隔离数据库**，检查 schema、任务数、结果身份、审计条目与租约恢复后再制定正式切换；禁止拿生产库做演练。对象存储语料发布物也需备份。备份保留期限、删除与审计保留由组织政策配置；当前代码未实现自动保留/清理作业。

本地 SQLite 适配器提供 `sqlite-backup` 和 `sqlite-restore` 命令，校验 SHA-256，恢复目标必须不存在；它的测试通过不能代替 PostgreSQL 恢复演练。真实 IdP、PostgreSQL 并发、数据库恢复、受信 CI、镜像运行和负载验收均需独立通过后才能对内部用户开放。

实现参考：[Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets)、[PostgreSQL pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html)、[GitHub OIDC 与可复用工作流](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/secure-your-work/security-harden-deployments/oidc-with-reusable-workflows)。
# 两类维护任务

`POST /internal/tasks` 的 `workflow_type` 支持 `upgrade_review`（默认，依赖 2.9.1 → 3.0.0）及 `configuration_change`（同为 2.9.1）。两个值属于同一应用任务体系，不是不同的应用版本。

配置审查必须提供 `base` 与 `head` 两个 Git 引用；服务端解析成不可变提交，核心检查为确定性工具，不支持 `generate_advice=true`。当前仅支持显式构造器 `drop_score` 数值变化，其他改动保留缺口。使用同一 Worker、权限校验、租约与取消机制；不会因浏览器读取任务状态而重新执行。

可信 CI 记录必须绑定该任务的候选提交、完整输入哈希、当前报告及目标依赖：升级为 3.0.0，配置为 2.9.1。继续人工评审需要独立审核者，分别说明静态风险或配置行为变化的接受理由；最新已接收的失败运行会使早先成功失效。CI 声明、静态报告和人工意见始终分开，不自动批准发布。
