"""Mechanical checks for explicitly stated numeric defaults; never a truth score."""
from __future__ import annotations
import re
import ast
import html
from src.paddleocr_retrieval_views import generation_evidence_text as evidence_text, module_for
from src.paddleocr_query_plan import claim_scope_module, explicit_versions

_DEFAULT=re.compile(r'\b([A-Za-z][\w]*)\b\s*(?:的)?\s*默认(?:值)?\s*(?:为|是|=|：|:)?\s*([-+]?\d+(?:\.\d+)?|True|False|None)\b',re.I)


def _defaults(text):
    pairs={}
    # Only a parsed function signature defines a Python default. A call example
    # or local assignment is an explicit choice, not a library default.
    try:
        tree=ast.parse(text)
    except (SyntaxError,ValueError):
        tree=None
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            arguments=node.args.posonlyargs+node.args.args
            values=list(zip(arguments[-len(node.args.defaults):],node.args.defaults)) if node.args.defaults else []
            values+=list(zip(node.args.kwonlyargs,node.args.kw_defaults))
            for arg,value in values:
                if isinstance(value,ast.Constant) and (value.value is None or type(value.value) in (int,float,bool)):
                    pairs.setdefault(arg.arg,set()).add(str(value.value).lower())
    column=None
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', text, re.S|re.I):
        cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip()
                 for c in re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', row, re.S|re.I)]
        if any('默认' in c for c in cells):
            column = next(i for i,c in enumerate(cells) if '默认' in c)
        elif column is not None and len(cells)>column and re.fullmatch(r'[A-Za-z][\w]*',cells[0]):
            value = cells[column]
            if re.fullmatch(r'[-+]?\d+(?:\.\d+)?|True|False|None',value,re.I):
                pairs.setdefault(cells[0],set()).add(value.lower())
    column=None
    for line in text.splitlines():
        cells=[c.strip().strip('`') for c in line.strip().strip('|').split('|')]
        if any('默认' in c for c in cells):
            column=next(i for i,c in enumerate(cells) if '默认' in c); continue
        if column is not None and len(cells)>column:
            symbol=re.fullmatch(r'[A-Za-z][\w]*',cells[0])
            value=cells[column]
            if symbol and re.fullmatch(r'[-+]?\d+(?:\.\d+)?|True|False|None',value,re.I):
                pairs.setdefault(symbol[0],set()).add(value.lower())
    return pairs


def check_mechanical_facts(claims:list[dict],evidence:list[dict],*,requirements:list[dict])->dict:
    by_id={h['chunk_id']:h for h in evidence}; requirement_status={}
    for req in requirements:
        symbols=req.get('symbols') or ([req['symbol']] if req.get('symbol') else [])
        applicable=[h for h in evidence if (not req.get('module') or h.get('module',module_for(h)) in (req['module'],'shared'))
                    and (not req.get('version') or h.get('version')==req['version'])]
        found=[h['chunk_id'] for h in applicable if all(s in evidence_text(h) for s in symbols)]
        requirement_status[req['id']]={'status':'APPLICABLE_EVIDENCE_FOUND' if found else 'MISSING_APPLICABLE_EVIDENCE',
                                       'evidence_ids':found,'assessment':'scope_and_symbol_only'}
    verdicts=[]
    for i,claim in enumerate(claims):
        assertions=list(_DEFAULT.finditer(claim['text'].replace('`','')))
        symbols={m[1] for m in assertions}
        claim_module=claim_scope_module(claim['text'],requirements)
        claim_versions=explicit_versions(claim['text'])
        relevant=[r for r in requirements if symbols.intersection(r.get('symbols') or [r.get('symbol')])
                  and (not claim_module or not r.get('module') or r['module']==claim_module)
                  and (not claim_versions or not r.get('version') or r['version'] in claim_versions)]
        if assertions and claim_module and not relevant:
            relevant=[{'module':claim_module,'version':None}]
        def applies(hit,req):
            return ((not req.get('module') or hit.get('module',module_for(hit)) in (req['module'],'shared'))
                    and (not claim_versions or hit.get('version') in claim_versions)
                    and (not req.get('version') or hit.get('version')==req['version']))
        cited=[by_id[cid] for cid in claim.get('evidence_ids',[]) if cid in by_id]
        missing_scope=(bool(claim_versions) and any(h.get('version') not in claim_versions for h in cited)) or any(not any(applies(h,r) for h in cited) for r in relevant)
        defaults={}
        for hit in cited:
            if claim_versions and hit.get('version') not in claim_versions:continue
            if relevant and not any(applies(hit,r) for r in relevant):continue
            for key,values in _defaults(evidence_text(hit)).items():defaults.setdefault(key,set()).update(values)
        contradicted=[m[1] for m in assertions if m[1] in defaults and m[2].lower() not in defaults[m[1]]]
        supported=[m[1] for m in assertions if defaults.get(m[1])=={m[2].lower()}]
        status='SCOPE_MISMATCH' if missing_scope else 'CONTRADICTED' if contradicted else 'CONSISTENT_PARSED_FACT' if assertions and len(supported)==len(assertions) else 'NOT_MECHANICALLY_CHECKABLE'
        verdicts.append({'claim_index':i,'status':status,'symbols':contradicted or supported})
    return {'assessment_type':'mechanical_scope_and_parsed_defaults_not_complete_answer_validation',
            'requirements':requirement_status,'claims':verdicts}
