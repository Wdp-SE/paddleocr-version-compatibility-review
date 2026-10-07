"""Bounded evidence investigation. Retrieved candidates never change static risks."""
from __future__ import annotations
import copy
import hashlib
import inspect
import json

VERSIONS=('v2.9.1','v3.0.0')

# These are source-grounded contract anchors, not a semantic correctness score.
_FACTS = {
    'legacy_ocr_result': {
        'v2.9.1': [{'path': 'paddleocr.py', 'tokens': ['ocr_res.append', 'zip(dt_boxes, rec_res)']}],
        'v3.0.0': [{'path': 'paddleocr/_pipelines/ocr.py', 'tokens': ['def ocr(', 'self.predict']},
                   {'path': 'docs/version3.x/pipeline_usage/OCR.md', 'tokens': ['rec_texts', 'rec_scores']}],
    },
    'removed_ppstructure': {
        'v2.9.1': [{'path': 'paddleocr.py', 'tokens': ['class PPStructure(']}],
        'v3.0.0': [{'path': 'docs/update/upgrade_notes.md', 'tokens': ['PPStructure', 'PPStructureV3', '移除']}],
    },
    'basic_ocr_surface': {
        'v2.9.1': [{'path': 'paddleocr.py', 'tokens': ['def ocr(']}],
        'v3.0.0': [{'path': 'paddleocr/_pipelines/ocr.py', 'tokens': ['def ocr(', 'self.predict']}],
    },
}


def _assess_content(check, version, rows):
    facts = check.get('required_facts', {}).get(version, [])
    tokens = check.get('required_tokens', {}).get(version, [])
    if not facts and tokens:
        facts = [{'tokens': tokens}]
    if not rows:
        return 'NO_VERIFIED_SOURCE', 'required_source_not_verified', []
    if not facts:
        return 'SOURCE_CANDIDATES_ONLY', 'content_criteria_not_defined', []
    missing = []
    for fact in facts:
        content = '\n'.join(r.get('content', '') for r in rows
                            if not fact.get('path') or r.get('document_path', r.get('path')) == fact['path'])
        absent = [token for token in fact['tokens'] if token not in content]
        if absent:
            missing.append({'path': fact.get('path'), 'tokens': absent})
    return ('INSUFFICIENT_CONTENT', 'content_requirements_not_covered', missing) if missing else (
        'REQUIREMENTS_COVERED', None, [])


def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build_checks(report:dict)->list[dict]:
    checks=[];seen=set()
    for finding in report.get('findings',[]):
        key=finding['rule_id']
        if key in seen:continue
        seen.add(key)
        checks.append({'check_id':f'check-{len(checks)+1}','query':finding['title']+' '+finding['desired_check'][:400],
                       'versions':list(VERSIONS),'finding_ids':[f['finding_id'] for f in report['findings'] if f['rule_id']==key],
                       'application':finding['application'],'required_tokens':{},
                       'required_facts':copy.deepcopy(_FACTS.get(key, {}))})
        # Component contracts have source anchors. Application regression
        # instructions are not facts the official SDK can establish.
        facts=checks[-1]['required_facts']
        if facts:
            checks[-1]['query']='PaddleOCR '+key+' '+ ' '.join(dict.fromkeys(
                value for rows in facts.values() for row in rows
                for value in [row['path'], *row['tokens']]))
    for gap in report.get('gaps',[]):
        if len(checks)>=12:break
        key=gap['code']
        if key in seen:continue
        seen.add(key)
        checks.append({'check_id':f'check-{len(checks)+1}','query':'PaddleOCR '+gap['detail'][:400],
                       'versions':list(VERSIONS),'gap_code':key,'required_tokens':{},
                       'evidence_domain':'application_validation'})
    if not checks:checks=[{'check_id':'check-1','query':'PaddleOCR 安装 推理 结果契约', 'versions':list(VERSIONS)}]
    return checks[:12]


def _bounded_method(method):
    try:
        parameters=inspect.signature(method).parameters
        return 'request_timeout' in parameters or any(p.kind==p.VAR_KEYWORD for p in parameters.values())
    except (ValueError,TypeError):return False


def investigate(checks:list[dict],gateway,*,clock,deadline_seconds:float=240,planner=None,
                resume:dict|None=None,cancelled=None,validate_evidence=None,on_progress=None)->dict:
    if not isinstance(checks,list) or not 1<=len(checks)<=12 or not 0<deadline_seconds<=240:
        raise ValueError('invalid investigation budget')
    for c in checks:
        if (not isinstance(c,dict) or not isinstance(c.get('check_id'),str) or not isinstance(c.get('query'),str)
                or not 0<len(c['query'])<=4000 or not isinstance(c.get('versions'),list)
                or not c['versions'] or any(v not in VERSIONS for v in c['versions'])):
            raise ValueError('invalid check scope')
    identity=_digest(checks)
    state=copy.deepcopy(resume) if resume else {'identity':identity,'completed':{},'search_calls':0,'planner_calls':0,'elapsed_seconds':0}
    if (state.get('identity')!=identity or type(state.get('search_calls')) is not int or not 0<=state['search_calls']<=24
            or type(state.get('planner_calls')) is not int or not 0<=state['planner_calls']<=3
            or not isinstance(state.get('completed'),dict) or not isinstance(state.get('elapsed_seconds'),(int,float))
            or not 0<=state['elapsed_seconds']<=240):
        raise ValueError('investigation resume identity or budget mismatch')
    started=clock(); prior_elapsed=state['elapsed_seconds']; trace=[]; checked={};unresolved=[]
    parents={c['check_id']:c for c in checks}
    if len(parents) != len(checks):
        # Repeated identical checks share one business obligation.
        if any(c != parents[c['check_id']] for c in checks):raise ValueError('ambiguous check identity')
    outcomes={}; accumulated={}
    stop='COMPLETED'; queue=list(parents.values()); seen_queries=set(); processed=0
    restored_plans=state.get('planned_checks', [])
    if not isinstance(restored_plans,list) or len(restored_plans)>12:raise ValueError('invalid saved plans')
    state['planned_checks']=[]
    def enqueue(parent_id, query, versions, suffix):
        parent=parents.get(parent_id)
        if (parent is None or not isinstance(query,str) or not 0<len(query)<=4000
                or not isinstance(versions,list) or not versions
                or any(v not in parent['versions'] for v in versions)):return
        proposal={**parent,'check_id':f'{parent_id}-{suffix}', 'parent_check_id':parent_id,
                  'query':query,'versions':list(dict.fromkeys(versions))}
        if any(r['query']==query and r['versions']==proposal['versions'] for r in queue):return
        queue.append(proposal);state['planned_checks'].append(proposal)
    for i,row in enumerate(restored_plans):
        if not isinstance(row,dict):raise ValueError('invalid saved plan')
        enqueue(row.get('parent_check_id'),row.get('query'),row.get('versions'),f'saved{i}')
    def unresolved_items():
        result=[]
        for parent_id,parent in parents.items():
            for version in parent['versions']:
                outcome=outcomes.get((parent_id,version),{})
                if outcome.get('status')!='REQUIREMENTS_COVERED':
                    result.append({'check_id':parent_id,'versions':[version],
                                   'reason':outcome.get('reason','ROUND_OR_ITEM_BUDGET'),
                                   'query':parent['query'], 'missing_facts':outcome.get('missing_facts',[])})
        return result
    def remaining():return max(0,deadline_seconds-prior_elapsed-(clock()-started))
    for round_no in range(1,4):
        batch=queue[:4];queue=queue[4:]
        if not batch:break
        for batch_index,check in enumerate(batch):
            processed+=1; parent_id=check.get('parent_check_id',check['check_id'])
            if check.get('evidence_domain')=='application_validation':
                for version in check['versions']:
                    outcomes[(parent_id,version)]={'check_id':parent_id,'version':version,
                        'status':'APPLICATION_VALIDATION_REQUIRED',
                        'reason':'application_validation_required','missing_facts':[], 'evidence_ids':[]}
                trace.append({'check_id':check['check_id'],'status':'APPLICATION_VALIDATION_REQUIRED',
                              'query':check['query'],'reason':'application_validation_required'})
                continue
            if on_progress:on_progress({'stage':'official_lookup','round':round_no,'check_id':check['check_id'],'processed':processed,'elapsed_seconds':round(prior_elapsed+clock()-started,2)})
            for version in dict.fromkeys(check['versions']):
                if cancelled and cancelled():stop='CANCELLED';break
                if remaining()<=0:stop='DEADLINE_EXCEEDED';break
                key=_digest({'query':' '.join(check['query'].casefold().split()),'version':version})
                if key in seen_queries:stop='DUPLICATE_QUERY';continue
                seen_queries.add(key)
                cached=state['completed'].get(key)
                if cached is None:
                    if not _bounded_method(gateway.search):stop='DEADLINE_NOT_ENFORCEABLE';break
                    if state['search_calls']>=24:stop='SEARCH_BUDGET_EXHAUSTED';break
                    state['search_calls']+=1
                    try:
                        cached=gateway.search(check['query'],version=version,language='zh',top_k=5,
                                              request_timeout=remaining(),retrieval_policy='paddleocr_evidence')
                        if not isinstance(cached,dict) or not isinstance(cached.get('results'),list):raise ValueError('invalid tool result')
                        state['completed'][key]=cached
                    except Exception as exc:
                        trace.append({'check_id':check['check_id'],'version':version,'status':'TOOL_FAILED',
                                      'reason':type(exc).__name__,'query':check['query']});continue
                good=[];verified_rows=[]
                for hit in cached.get('results',[])[:20]:
                    try:
                        if validate_evidence is None:raise ValueError('source validator not configured')
                        if hit.get('version')!=version:raise ValueError('wrong version')
                        original=validate_evidence(hit,remaining())
                        if not isinstance(original,dict) or original.get('chunk_id')!=hit.get('chunk_id'):raise ValueError('identity')
                        checked[original['chunk_id']]=original;good.append(original['chunk_id'])
                        verified_rows.append(original)
                    except Exception:
                        trace.append({'check_id':check['check_id'],'version':version,'status':'EVIDENCE_VALIDATION_REQUIRED',
                                      'query':check['query'],'evidence_id':hit.get('chunk_id')})
                evidence = accumulated.setdefault((parent_id,version),{})
                evidence.update({row['chunk_id']:row for row in verified_rows})
                good=list(evidence)
                assessment,reason,missing_facts=_assess_content(parents[parent_id],version,list(evidence.values()))
                outcome={'check_id':parent_id,'version':version,'status':assessment,'reason':reason,
                         'missing_facts':missing_facts,'evidence_ids':good}
                if outcomes.get((parent_id,version),{}).get('status')!='REQUIREMENTS_COVERED':
                    outcomes[(parent_id,version)]=outcome
                trace.append({'check_id':check['check_id'],'version':version,'query':check['query'],
                              'status':'SOURCE_VERIFIED_CANDIDATES' if good else 'NO_VERIFIED_EVIDENCE',
                              'content_status':assessment, 'missing_facts':missing_facts,
                              'evidence_ids':good,'cached':key in (resume or {}).get('completed',{})})
            if stop in ('CANCELLED','DEADLINE_EXCEEDED','DEADLINE_NOT_ENFORCEABLE','SEARCH_BUDGET_EXHAUSTED'):
                queue=batch[batch_index+1:]+queue
                break
        if stop in ('CANCELLED','DEADLINE_EXCEEDED','DEADLINE_NOT_ENFORCEABLE','SEARCH_BUDGET_EXHAUSTED'):break
        unresolved=unresolved_items()
        replannable=[r for r in unresolved if r['reason'] in ('required_source_not_verified','content_requirements_not_covered')]
        if planner and replannable and remaining()>0 and state['planner_calls']<3 and processed+len(queue)<12 and round_no<3:
            try:
                if not _bounded_method(planner):raise ValueError('unbounded planner')
                state['planner_calls']+=1
                proposal=planner(unresolved=copy.deepcopy(replannable),request_timeout=remaining())
                if not isinstance(proposal,list):raise ValueError('invalid planner result')
                for row in proposal[:min(4,12-processed-len(queue))]:
                    if isinstance(row,dict):enqueue(row.get('parent_check_id'),row.get('query'),row.get('versions'),f'r{round_no}')
                trace.append({'status':'PLANNER_COMPLETED','round':round_no,'proposed_count':len(proposal)})
            except Exception:
                trace.append({'status':'PLANNER_FAILED','round':round_no})
    unresolved=unresolved_items()
    state['elapsed_seconds']=min(240,max(0,prior_elapsed+clock()-started))
    if unresolved and stop=='COMPLETED':stop='NO_NEW_EVIDENCE'
    return {'schema_version':1,'stop_reason':stop,'trace':trace,'checked_evidence':list(checked.values()),
            'check_assessments':list(outcomes.values()),
            'unresolved':unresolved,'search_calls':state['search_calls'],'planner_calls':state['planner_calls'],
            'elapsed_seconds':state['elapsed_seconds'],'resume':state,
            'assessment':'source_verified_candidates_not_confirmed_business_impact'}
