"""Editable variants of one original application, not certified upgrades."""
from services.paddleocr_application_inputs import load_original_application, parse_application_inputs

DEMO_CASES = {
    '升级风险：旧结果消费': '定位调用层返回的原始结果与归一化层旧式索引，沿应用关系整理下游 JSON 回归要求。',
    '修改后复查：字段适配': '按目标版字段读取文本与得分；没有发现旧式索引也不能认定升级成功，环境和真实文档回归仍待完成。',
    '资料缺口：调用层未提交': '调用层缺失，无法证明结果来源与影响关系；报告应明确补充文件要求，不能给出无影响结论。',
}


def demo_application(label):
    if label not in DEMO_CASES:
        raise ValueError('unknown interview example')
    rows = load_original_application()
    old = '''def normalize_legacy(raw):
    return [{'text': line[1][0], 'confidence': line[1][1], 'page_index': page_index}
            for page_index, page in enumerate(raw) if page is not None for line in page]
'''
    new = '''from result_adapter import normalize_v3


def normalize_legacy(raw):
    return normalize_v3(raw)
'''
    # Explicitly marked demonstration variants. The original manifest/files are
    # unchanged and runtime regression history is never attached automatically.
    for row in rows:
        if row['path'] == 'application/normalizer.py':
            row['content'] = new if label == '修改后复查：字段适配' else old
        elif row['path'] == 'application/design.md':
            row['content'] += '\n当前输入为学生原创应用的可编辑演示变体，不附带运行成功证明。扫描页经 OCR、归一化后由下游消费 JSON。\n'
    if label == '资料缺口：调用层未提交':
        rows = [r for r in rows if r['path'] != 'application/ocr_client.py']
    return parse_application_inputs(rows)
