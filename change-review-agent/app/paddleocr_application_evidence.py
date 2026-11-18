"""Validate application declarations separately from official version evidence."""
import hashlib
import json
from pathlib import PurePosixPath


def validate_contracts(report, files):
    expected=[]
    for path,content in files.items():
        if PurePosixPath(path).suffix.casefold()!='.json' or 'contract' not in PurePosixPath(path).name.casefold():
            continue
        try:
            original=json.loads(content)
            if not isinstance(original,dict):raise ValueError()
        except (ValueError,RecursionError):
            if not any(g.get('code')=='invalid_output_contract' and g.get('application',{}).get('path')==path for g in report.get('gaps',[])):
                raise ValueError('无效应用契约缺少明确缺口') from None
            continue
        requirements={}
        for key in ('required_line_fields','output_fields','page_fields'):
            values=original.get(key)
            if (isinstance(values,list) and 1<=len(values)<=32
                    and all(isinstance(v,str) and 1<=len(v)<=80 for v in values) and len(set(values))==len(values)):
                requirements[key]=values
            elif key in original and not any(g.get('code')=='invalid_contract_requirement' and g.get('application',{}).get('path')==path for g in report.get('gaps',[])):
                raise ValueError('应用契约字段无效但未记录缺口')
        if original.get('page_policy') in ('preserve_empty_pages','omit_empty_pages'):
            requirements['page_policy']=original['page_policy']
        elif 'page_policy' in original and not any(g.get('code')=='unsupported_contract_requirement' and g.get('application',{}).get('path')==path for g in report.get('gaps',[])):
            raise ValueError('应用契约分页策略无效但未记录缺口')
        expected.append({'path':path,'sha256':hashlib.sha256(content.encode()).hexdigest(),
            'line_start':1,'line_end':len(content.splitlines()),'source_identity':'submitted_application',
            'requirements':requirements,'status':'DECLARED_NOT_VERIFIED','runtime_verified':False})
    rows = report.get('application_contracts', [])
    if not isinstance(rows, list) or len(rows) > 12 or rows!=expected:
        raise ValueError('应用契约清单、字段或来源不完整')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('应用契约来源无效')
        content = files.get(row.get('path'))
        requirements = row.get('requirements')
        if (content is None or row.get('path') in seen
                or row.get('sha256') != hashlib.sha256(content.encode()).hexdigest()
                or row.get('source_identity') != 'submitted_application'
                or row.get('runtime_verified') is not False
                or type(row.get('line_start')) is not int or row['line_start'] != 1
                or type(row.get('line_end')) is not int or row['line_end'] != len(content.splitlines())
                or not isinstance(requirements, dict)
                or set(requirements) - {'required_line_fields', 'output_fields', 'page_fields', 'page_policy'}):
            raise ValueError('应用契约未绑定本次提交的原文')
        try:
            original = json.loads(content)
        except (ValueError, RecursionError):
            raise ValueError('应用契约原文无法解析') from None
        if not isinstance(original, dict) or any(original.get(k) != v for k, v in requirements.items()):
            raise ValueError('应用契约要求与原文不一致')
        seen.add(row['path'])
    expected_regressions=[]
    for source in expected:
        req=source['requirements']
        checks=[f'{k}: '+', '.join(v) for k,v in req.items() if isinstance(v,list)]
        if 'page_policy' in req:checks.append('分页策略: '+req['page_policy'])
        if checks:
            expected_regressions.append({'kind':'application_contract','rule_id':'application_output_contract',
                'finding_id':'contract-'+hashlib.sha256((source['path']+source['sha256']).encode()).hexdigest()[:16],
                'application':{'path':source['path'],'line':1},'source':source,
                'input_conditions':['带文本页','空页','多页','非法字段或长度不匹配'],
                'expected_contract':'按本次提交的应用输出契约验证；测试定义不代表测试已经执行。',
                'checks':checks,'status':'NOT_EXECUTED','runtime_verified':False})
    actual=[r for r in report.get('regression_requirements',[]) if r.get('kind')=='application_contract']
    if actual!=expected_regressions:
        raise ValueError('应用契约回归要求不完整或冒充已执行')
