"""Review facts for visible demos; additional probes remain separate from tuning.

Facts guide source and answer review, never feed generation or retrieval ranking.
"""
FACTS = [
    ['初始化 use_angle_cls=True 加载分类模型', 'ocr(cls=True) 控制调用时启用分类，两处作用不同'],
    ['已有裁剪图片关闭检测 det=False', '保留识别 rec=True 或默认识别，不应 rec=False'],
    ['line[1][0] 为文本', 'line[1][1] 为识别得分'],
    ['drop_score 过滤低于阈值的识别输出', '默认值 0.5；不是识别模型精度保证'],
    ['rec_batch_num 是识别阶段同时前向的图片数', '不能推断为应用文档页数；本题不要求默认值'],
    ['新版使用 predict 输入', 'ocr 入口仍保留并委托 predict；入口存在不代表结果消费兼容'],
    ['rec_texts 是过滤后的识别文本', 'rec_scores 是对应得分，需保持对应关系'],
    ['text_rec_score_thresh 影响识别结果保留', '不能由过滤门槛推断业务准确率改善'],
    ['None 使用初始化配置', 'False 显式关闭文本行方向分类；适用通用 OCR 产线'],
    ['结果对象支持 save_to_json', 'SDK JSON 不证明本应用输出契约，后者需应用资料'],
    ['旧版嵌套列表中的文本/得分索引', '新版结果对象字段与返回容器应区分', '旧读取方式需适配，不能称全部调用入口失效'],
    ['use_angle_cls 迁移为 use_textline_orientation', '两版方向分类能力有区别，不保证等效结果'],
    ['3.x 日志机制调整，show_log 的接受状态需查目标版实现，不能凭文档未提及就断言移除',
     'det_db_thresh 对应 text_det_thresh；不是所有旧参数都直接删除'],
    ['分别检查入口、参数与结果读取', '需要两版来源，不可仅凭目标版文档断言全部兼容'],
    ['官方知识库不能证明未提交的应用 JSON 契约', '需调用/归一化/消费者/契约和实际回归资料'],
]

PROBES = [
    ('v2.9.1', '扫描件中低分文字没有进入 JSON，PaddleOCR 2.9.1 的 drop_score 可能起什么作用？'),
    ('v3.0.0', 'OCR 3.0.0 结果中的文本和置信度应分别读取哪些字段，怎样避免错配？'),
    ('v2.9.1', 'PaddleOCR 2.9.1 对已有单行裁剪图片关闭检测后，是否还需要保留识别？'),
    ('all', '没有提交应用契约或测试，仅看 PaddleOCR 官方资料能保证我们的页号和 JSON 不变吗？'),
]
