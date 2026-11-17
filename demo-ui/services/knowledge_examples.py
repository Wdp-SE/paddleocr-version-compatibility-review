"""Version-scoped maintenance questions for the same document-processing app.

These are editable queries, never cached answers or compatibility verdicts.
"""
from dataclasses import dataclass

from services.internal_application_scope import CURRENT_DEPENDENCY, TARGET_DEPENDENCY


@dataclass(frozen=True)
class KnowledgeExample:
    scope: str
    category: str
    question: str
    purpose: str


EXAMPLES = (
    KnowledgeExample(CURRENT_DEPENDENCY, '调用接入',
        '扫描页有180度旋转文字时，PaddleOCR 2.9.1 的 use_angle_cls 与 ocr(cls=True) 如何配合？',
        '核对方向分类器的初始化开关与调用开关，避免以为只改其中一处就会生效。'),
    KnowledgeExample(CURRENT_DEPENDENCY, '识别链路',
        'PaddleOCR 2.9.1 对已经裁剪好的文字图片，如何只做识别而不做检测？',
        '区分整页检测与已裁剪文字识别，避免调用模式与输入不匹配。'),
    KnowledgeExample(CURRENT_DEPENDENCY, '结果归一化',
        'PaddleOCR 2.9.1 的 ocr() 返回嵌套列表时，line[1][0] 和 line[1][1] 各取到什么？',
        '为应用归一化层找到旧版结果读取依据。'),
    KnowledgeExample(CURRENT_DEPENDENCY, '识别过滤',
        'PaddleOCR 2.9.1 的 drop_score 参数如何过滤低置信度识别结果，默认值是多少？',
        '排查低分文字未进入后续输出的可能原因，不把过滤误当识别失败。'),
    KnowledgeExample(CURRENT_DEPENDENCY, '批量处理',
        'PaddleOCR 2.9.1 的 rec_batch_num 控制什么，是否等于一次处理的扫描页数？',
        '区分识别批量大小与应用的文档分页策略。'),
    KnowledgeExample(TARGET_DEPENDENCY, '调用迁移',
        '接入 PaddleOCR 3.0.0 通用 OCR 时，predict() 如何调用，ocr() 入口还保留吗？',
        '确认目标版本调用入口，避免仅凭名称变化判断整条链路兼容。'),
    KnowledgeExample(TARGET_DEPENDENCY, '结果归一化',
        'PaddleOCR 3.0.0 通用 OCR 的 rec_texts 和 rec_scores 分别表示什么，如何对应？',
        '核对新版文本与分数的读取方式，为适配层迁移提供依据。'),
    KnowledgeExample(TARGET_DEPENDENCY, '识别过滤',
        'PaddleOCR 3.0.0 的 text_rec_score_thresh 如何影响保留下来的识别结果？',
        '确认目标版分数阈值语义，排查文字缺失而非直接断言模型漏识别。'),
    KnowledgeExample(TARGET_DEPENDENCY, '方向处理',
        'PaddleOCR 3.0.0 通用 OCR 的 use_textline_orientation 传 None 时如何处理？如何显式关闭文本行方向分类？',
        '核对 None 与 False 的区别，避免误以为省略参数就会关闭方向处理。'),
    KnowledgeExample(TARGET_DEPENDENCY, '结果导出',
        'PaddleOCR 3.0.0 的 OCR 结果如何保存为 JSON？这是否等于应用已有 JSON 契约？',
        '区分 SDK 导出格式与应用自己的输出字段；内部契约仍需实际应用资料。'),
    KnowledgeExample('all', '结果迁移',
        'PaddleOCR 2.9.1 升级到 3.0.0，为什么旧代码按 line[1][0] 和 line[1][1] 读取结果需要调整？',
        '对照两版结果契约；要定位本应用哪一行受影响，应提交代码发起升级审查。'),
    KnowledgeExample('all', '方向参数迁移',
        'PaddleOCR 2.9.1 的 use_angle_cls 和 3.0.0 的 use_textline_orientation 如何衔接？',
        '对照方向分类参数迁移，核查调用封装是否仍使用旧参数。'),
    KnowledgeExample('all', '参数迁移',
        'PaddleOCR 2.9.1 升级到 3.0.0 后，show_log 和 det_db_thresh 这些旧参数应如何处理？',
        '识别移除参数与重命名参数的区别，避免直接复制旧初始化配置。'),
    KnowledgeExample('all', '升级准备',
        '文档处理应用准备从 PaddleOCR 2.9.1 升级到 3.0.0，调用入口、参数和结果读取应分别核查什么？',
        '生成升级前的组件知识核查清单；具体代码风险由同一应用的审查入口定位。'),
    KnowledgeExample('all', '证据边界',
        '仅凭 PaddleOCR 2.9.1 和 3.0.0 官方资料，能否确认我们应用升级后下游 JSON 兼容？',
        '展示证据边界：官方组件说明不能代替应用契约、调用代码和真实回归记录。'),
)


def examples_for_scope(scope):
    return tuple(row for row in EXAMPLES if row.scope == scope)


def sync_scope_examples(state, scope):
    """Replace stale canned text on scope changes, preserving a custom draft."""
    rows = examples_for_scope(scope)
    if not rows:
        return
    questions = {row.question for row in rows}
    if state.get('official_example') in questions:
        return
    old = state.get('official_example')
    question = state.get('official_question', '')
    state['official_example'] = rows[0].question
    if not question.strip() or question == old:
        state['official_question'] = rows[0].question
    state.pop('official_result', None)
    state.pop('official_result_top_k', None)
