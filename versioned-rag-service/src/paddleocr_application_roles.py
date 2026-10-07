"""File roles describe analysis scope, never runtime compatibility."""
from pathlib import PurePosixPath
import re


def application_role(path: str, content: str) -> str:
    name = PurePosixPath(path).name.casefold()
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix == '.md':
        return 'design_document'
    if suffix == '.json' and 'contract' in name:
        return 'output_contract'
    if name.startswith('requirements') or name in ('pyproject.toml', 'setup.cfg'):
        return 'dependency_manifest'
    if suffix in ('.yaml', '.yml', '.json'):
        return 'runtime_configuration'
    if suffix == '.py':
        if name.startswith('test_') or '/tests/' in '/' + path:
            return 'application_test'
        if re.search(r'\b(?:from\s+paddleocr\b|import\s+paddleocr\b)', content):
            return 'sdk_call_code'
        return 'application_code'
    return 'supporting_document'


def regression_requirements(findings: list[dict]) -> list[dict]:
    rows = []
    for finding in findings:
        rule = finding['rule_id']
        layout = rule == 'removed_ppstructure'
        rows.append({
            'finding_id': finding['finding_id'], 'rule_id': rule,
            'application': finding['application'],
            'input_conditions': ['含表格与段落的版面页', '空页', '多页'] if layout else ['带文本页', '空页', '多页'],
            'expected_contract': finding['desired_check'],
            'checks': (['替代接口初始化及调用', '区域/表格/OCR字段映射', '下游输出契约'] if layout else
                       ['rec_texts与rec_scores对应关系', '字段类型及顺序', '下游JSON契约'] if rule == 'legacy_ocr_result' else
                       ['实际依赖与模型加载', '基本调用执行', '下游归一化及JSON契约']),
            'status': 'NOT_EXECUTED', 'runtime_verified': False,
        })
    return rows
