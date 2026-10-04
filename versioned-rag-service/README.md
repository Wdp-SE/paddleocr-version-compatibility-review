# PaddleOCR 固定版本知识服务

活动范围是 PaddlePaddle/PaddleOCR 的 v2.9.1 与 v3.0.0 中文 OCR、版面/表格解析、安装部署与升级资料，以及必要的官方代码和配置契约。默认查询最新**已收录** v3.0.0，不宣称它是上游当前最新版。每条来源绑定 tag、提交、路径和原始文件 SHA-256，分块保留来源行号及代码/表格边界。

## API

- `GET /health`：服务、生成配置、活动工作区和构建指纹。
- `GET /public/workspace`：来源与版本、语料量、兼容性检查范围。
- `GET /public/documents`、`POST /public/document`：固定来源与原文片段；后者接受已登记 source_id 或 document_id。
- `POST /public/search`：只读检索，按版本过滤，最终证据 Top-K 1–20。
- `POST /public/query`：结构化生成，逐条校验引用成员；失效、证据不足和模型错误分别返回具体状态。
- `POST /public/compatibility-review`：审查受限应用文本的 v2.9.1→v3.0.0 静态兼容性，返回应用位置、版本证据、风险和未知项。不执行文本。
- `POST /public/review-advice`：仅用明确提交的当前目标版本证据生成核查建议；模型不批准升级。

```json
{
  "source_version": "v2.9.1",
  "target_version": "v3.0.0",
  "files": [{"path": "app.py", "content": "from paddleocr import PPStructure\nengine = PPStructure()\n"}]
}
```

工具限制最多 12 个文件、单文件 50 KB、总计 200 KB；文件名是安全的显示标签，不作为服务器路径读取。AST 复杂度受限，动态反射、不支持的版本和缺失契约不能得到虚假“安全”结论。所有报告标记 `runtime_verified=false`，真正升级还需要原始文档集的推理和下游契约回归。

## 来源构建与指标

构建配置在 `config/paddleocr_project.json`、`config/paddleocr_source_selection.json`，构建器在 `scripts/build_paddleocr_corpus.py`，实际来源与排除范围在活动语料的 `SOURCE_AUDIT.md`。v3 tag 中的 version2.x 教程排除在新接口问答外。只复制获许可的相关文本，不搬运链接中的权重、图像、视频和数据集。

默认 BM25 保留为可解释基线；任务自适应重排只是一条可选择、失败可回退的实验链路，不能因使用大模型就宣称准确率更高。新语料的来源召回与静态案例验收独立记录，旧业务成绩不作为当前成绩。

```powershell
python -m uvicorn src.public_server:app --host 127.0.0.1 --port 8770
python -m pytest tests/test_paddleocr_corpus.py tests/test_paddleocr_compatibility.py tests/test_paddleocr_release.py -q
```

生成使用服务端 DeepSeek 环境变量；不在 Streamlit secrets 中放模型密钥。公网模式固定活动语料，不接受旧领域环境覆盖。本轮本地实现不代表已执行线上发布。
