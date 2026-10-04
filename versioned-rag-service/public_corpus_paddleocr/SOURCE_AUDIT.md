# PaddleOCR 固定版本语料来源审计

PaddleOCR 官方中文资料与代码/配置辅助证据，固定至 v2.9.1 和 v3.0.0 两个提交；默认查询最新已收录版本 v3.0.0。

收录 58 个版本化文件：44 篇中文 Markdown、9 个 Python 契约、5 个 YAML 配置。代码与 YAML 单独计数。

Markdown 保留表格、代码块；Python 使用 AST 定义边界；YAML 保留顶层映射。块长度为软上限，完整大表格或函数可以超过它。每个块内容是原文行切片。

两个 tag 的提交由 GitHub Git ref API 验证。原始字节 SHA-256 以及固定 Git tree 的 blob SHA-1 双重校验；tag 元数据、完整树和许可正文保存在 provenance/。

Apache-2.0；PaddlePaddle 为原作者。文档中的图片、模型和数据链接仅保留文字，不下载媒体、权重、数据。v3.0.0 的历史 docs/version2.x 不收录，不能借发布提交误判旧 API 的适用版本。

公开运行默认 BM25。更大规模的 BGE 与 cross-encoder 重排实验依赖单独下载的模型资产，未随公网服务部署，不能作为线上成绩展示；只有原文、代码、数据和模型身份均匹配且通过留出门槛的报告才可改变默认策略。索引中的 dense_vectors 是字符哈希基线。144 个图片引用仍未提取，不能把链接当作图像知识或声称已经运行 OCR。

| 版本 | 类型 | 官方固定提交文件 | 原始字节 SHA-256 |
| --- | --- | --- | --- |
| v2.9.1 | markdown | [PaddleOCR 项目说明](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/README.md) | `22408f3d42741b88dec6913e6ad1277c21bbc9ac230ae963e4b69d218a62c595` |
| v2.9.1 | python | [PaddleOCR __init__ 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/__init__.py) | `1b031dc1a998ffaf562e4ce0b948b19533509bba18f4608bd48ad5ff7ca5a3fd` |
| v2.9.1 | yaml | [PaddleOCR ch_PP-OCRv4_det_student 模型配置](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml) | `f233f06e1c67d2cd2a20361be7444dbbafd93288c1191cbd089b5a1bca8b4a53` |
| v2.9.1 | yaml | [PaddleOCR ch_PP-OCRv4_rec 模型配置](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml) | `01afd2f8fb272e3f4f1722b00ff4f4df43bb5a71be8d1d9593b85d9ff2fb4595` |
| v2.9.1 | yaml | [PaddleOCR SLANet_ch 模型配置](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/configs/table/SLANet_ch.yml) | `9933986765695c2588765597b60e89ded84c75893457c9d52661c15291ad4a97` |
| v2.9.1 | markdown | [OCR 常见问题](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/FAQ.md) | `b50c9d0955f1a81038285cad126462d49e1b626d0c98f5304afd86506694b6e2` |
| v2.9.1 | markdown | [文本方向分类](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/angle_class.md) | `23d537b3e5f1094915410ede9cea8a9a6df7d8bbac6a6a0b5ec7ffdeb2dbd3b8` |
| v2.9.1 | markdown | [OCR 配置说明](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/config.md) | `74a506f54426d692b60361ca01660ce6683ebdd8a2f7e7a287ebb478ac86a38b` |
| v2.9.1 | markdown | [OCR 推理参数](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/inference_args.md) | `60812c30e0cd0d9ea351b82b3df0f54106cc36f3943f2072b88dd2a3fb6d59af` |
| v2.9.1 | markdown | [OCR 模型推理](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/inference_ppocr.md) | `54e49679a1f0303105a93b7d23665bb92b2d04af7ada32d89c6bbc8e934fc57e` |
| v2.9.1 | markdown | [PaddleOCR 安装依赖](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/installation.md) | `05315110378d6df377a839df3928c1867210b5d79440ceb1858ca14084fe8da0` |
| v2.9.1 | markdown | [OCR 多语言使用](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/multi_languages.md) | `f2b88a24c9265a73372a4e2b93e0a9a13deebfb4eb65425af2e75eed260d8ad1` |
| v2.9.1 | markdown | [OCR 或结构化解析快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/quickstart.md) | `17605d9dfafc66aab465cbf73f91c8c8800bc389fef4fec4925fe1076a72e981` |
| v2.9.1 | markdown | [表格识别使用](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/table_recognition.md) | `2c47abfbd32ecac942363c745d0d6ee88a0b1c5431780dfa2ad5e4efc569d6c0` |
| v2.9.1 | markdown | [OCR 结果可视化](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/visualization.md) | `c9c89704f7cc948ef721ca49c52b1de8cdedc814d128e90d9e1e9872afd59abd` |
| v2.9.1 | markdown | [Python 包安装与 OCR 结果用法](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/whl.md) | `9bc1fb30acf1da3ef10613fc44b8748f70ac7fb17d5b449b281fa49d869e455a` |
| v2.9.1 | python | [PaddleOCR paddleocr 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/paddleocr.py) | `37da5a58692bcd37c8d9cf60765c449fa3662de82d416f04358bd8ead87e7162` |
| v2.9.1 | markdown | [PP-Structure 中文说明](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/README_ch.md) | `702fe8dbe6130cf224c6c88ff58fe293e320d81f944fc80e938acf792bfb55e0` |
| v2.9.1 | markdown | [PP-Structure 推理与结果格式](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/docs/inference.md) | `cc0532b36a380b0e75029e8441fd4f9eb687d34fa0c6a09a2ac541b381ab6ad4` |
| v2.9.1 | markdown | [OCR 或结构化解析快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/docs/quickstart.md) | `519f0752788fcae2888970f48f433edd82e1e3653284ea8274c40c456af368bc` |
| v2.9.1 | markdown | [PP-Structure 中文说明](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/layout/README_ch.md) | `037db40341ebaa6921b41a8ff505a177224731416ad1ea477a755c3815fcf749` |
| v2.9.1 | python | [PaddleOCR predict_system 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/predict_system.py) | `e8bce2339ec2058f19372924dbff03707a4ffe37d7732fcadfe349b9943186c8` |
| v2.9.1 | markdown | [PP-Structure 中文说明](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/recovery/README_ch.md) | `09ba2bc597d469f9996d45fca956890bca684f7320814a06a229901ecbc0cf91` |
| v2.9.1 | markdown | [返回文字位置](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/return_word_pos.md) | `5f64b30b4accf28e2f6b9acc030ec3819d9d26232d21247db778409df4a237ff` |
| v2.9.1 | markdown | [PP-Structure 中文说明](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/ppstructure/table/README_ch.md) | `0cf25f375eb9ec3ee4dac8c5433ae7ba9236b2dee629309ae35f6fdff76b4b54` |
| v3.0.0 | markdown | [PaddleOCR 项目说明](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/README.md) | `b2d88b0efe84623dff83d13a8dc208b032eb9c1be2e03b6f4fcdbb4dcea54e08` |
| v3.0.0 | yaml | [PaddleOCR SLANeXt_wired 模型配置](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/configs/table/SLANeXt_wired.yml) | `0b7895ae6b73e205748ff6bef8db0906768281e5591839ba61e3d21fb49e39ed` |
| v3.0.0 | yaml | [PaddleOCR SLANet_plus 模型配置](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/configs/table/SLANet_plus.yml) | `5eb686fc725998a40513cc3222d09fd18d51179a30b1523934374f5ab1fb8a5c` |
| v3.0.0 | markdown | [PaddleOCR 快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/quick_start.md) | `54344c131077a19b8d81912d968e0098705e4ed6b83633024c07082e1996bc5b` |
| v3.0.0 | markdown | [PaddleOCR 3.0 升级说明](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/update/upgrade_notes.md) | `7b49f5758e45b217b8c3ed04aea3913e26fbf1d1d1a26277d09e55ed365fd994` |
| v3.0.0 | markdown | [高性能推理部署](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/deployment/high_performance_inference.md) | `e1ea0614aa091df2e8a818248836cbc7fc3bf11bff167fad327f0b9cb1e343b0` |
| v3.0.0 | markdown | [服务化部署](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/deployment/serving.md) | `19805ff5ea655fe68975296650e9429698ecd6010f33e484b5170c643b368556` |
| v3.0.0 | markdown | [PaddleOCR 安装依赖](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/installation.md) | `1ac79a8903314d5a0da69c5664fb04755f51a3beb961936a77662c0c15dd5e5b` |
| v3.0.0 | markdown | [PaddleOCR 日志配置](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/logging.md) | `f96479622438d62f5f6ff478d12a709c8436d5fd23a24e801c597a944e9d7c84` |
| v3.0.0 | markdown | [文档图像方向分类模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/doc_img_orientation_classification.md) | `1afd973dbc4999b600745be7f5fc27f0db78c6d5201b1e7da1746b56e2574d12` |
| v3.0.0 | markdown | [版面检测模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/layout_detection.md) | `4567d95d8e4571f682b3210d3441c2e661f73a528d321857dbf81972e72d6fbd` |
| v3.0.0 | markdown | [表格单元格检测模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/table_cells_detection.md) | `47d4ee3527d14c0eef4e71382fbf3c454a8fcb6a95a55a071702891edcd5ba58` |
| v3.0.0 | markdown | [表格分类模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/table_classification.md) | `d1234c970180351c216c3be72ad43b8eeb3ef0997f309766c65475e0c1777d0d` |
| v3.0.0 | markdown | [表格结构识别模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/table_structure_recognition.md) | `5742979908ca42ff2af9c07adac5e10b0cecabf64111799a6245249761b152b0` |
| v3.0.0 | markdown | [文本检测模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/text_detection.md) | `cc141826eb569a7351f5221bc335d6eb686cd55a56bef32ab9e2aadd23b250af` |
| v3.0.0 | markdown | [文本图像矫正模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/text_image_unwarping.md) | `387913b8d84b3fdc3a0dfe80a771e8cc0c206da75074abc050ac04c3f93630d6` |
| v3.0.0 | markdown | [文本行方向分类模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/text_line_orientation_classification.md) | `5604bc5726d3201792ad41fda5521ef597e9bee399058e3d4640069e2c63f819` |
| v3.0.0 | markdown | [文本识别模块](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/module_usage/text_recognition.md) | `585e1897b961f61e5735b8baf1589320b51d5e4f2b6552330419a99174ddf730` |
| v3.0.0 | markdown | [PaddleOCR 与 PaddleX 的接口关系](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/paddleocr_and_paddlex.md) | `01c481745d6772b8fbdc6bf4c653611f6a51d3fff4ec09b37bfa0c6aae1d5dd5` |
| v3.0.0 | markdown | [通用 OCR 产线使用与结果格式](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/pipeline_usage/OCR.md) | `6ce2310fc0b471e634a57fb54718fa56d79a00867515b3bf5be397163ab8927a` |
| v3.0.0 | markdown | [PP-StructureV3 产线使用与结果格式](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/pipeline_usage/PP-StructureV3.md) | `edf596dfd26a6a0814650495f2b503933822b06230cf045a3b4b2acbce72950d` |
| v3.0.0 | markdown | [文档图像预处理产线](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/pipeline_usage/doc_preprocessor.md) | `2aeea097fd70f5e9e65257deb2539626894691a56e5f88286fb9091a4313a557` |
| v3.0.0 | markdown | [表格识别产线](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/pipeline_usage/table_recognition_v2.md) | `6653f429a4f29e9867235eaaa1162f43c3ba7733e2d2991f4b9de22356b01ba5` |
| v3.0.0 | python | [PaddleOCR __init__ 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/paddleocr/__init__.py) | `3cb0d038cfa69a096dea052aceabdc250032c011211afa8a12b9ac80244f3642` |
| v3.0.0 | python | [PaddleOCR _common_args 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/paddleocr/_common_args.py) | `dd33c30e4c6b98ce14a87574df561bdd36f627bdcd90f65d31db60422d94dcdf` |
| v3.0.0 | python | [PaddleOCR __init__ 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/paddleocr/_pipelines/__init__.py) | `aab3bf7b7438cbd4159e5a3b1fd52ad760f313b1afec33c9cd582642b3e3f1b7` |
| v3.0.0 | python | [PaddleOCR base 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/paddleocr/_pipelines/base.py) | `8f9ace27768cf162bc322d37cdd0fae1161f06f66abe75f56821f385a7fe859e` |
| v3.0.0 | python | [PaddleOCR ocr 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/paddleocr/_pipelines/ocr.py) | `d81073c98ac08125c389d8a292837d9b2e12242ba0601ef65fb5d1c5a176a9a8` |
| v3.0.0 | python | [PaddleOCR pp_structurev3 公共 API 实现](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/paddleocr/_pipelines/pp_structurev3.py) | `9281167b47940f4cc81fe1f37e934b658981483b153021d68b07fab38c0b0239` |
| v2.9.1 | markdown | [模型推理总览](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/inference.md) | `95331d6f75b8cefd5fdf659b9260e328fbee1e5b8dddbf05f70c8480e0e82d36` |
| v2.9.1 | markdown | [算法推理与模型导出](https://github.com/PaddlePaddle/PaddleOCR/blob/07603421c20a96bb94bb87d0c4211032527ae706/doc/doc_ch/algorithm_inference.md) | `5471b532b2308b3e4fb4bf7f84ba6b29ac6850b57f154df10bfbc17f6fedb35c` |
| v3.0.0 | markdown | [获取 ONNX 模型](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/deployment/obtaining_onnx_models.md) | `d5374680da2c5c70653959b088222a08acbded2d72babaefadead050d6e797e1` |
| v3.0.0 | markdown | [端侧部署](https://github.com/PaddlePaddle/PaddleOCR/blob/a8474288ad53c0f439c272b786c5fa6240f0cf27/docs/version3.x/deployment/on_device_deployment.md) | `e4b6e90aa8ae55b99795a576f6888e693bf9d25b31f98478693a0c240d668e3e` |
