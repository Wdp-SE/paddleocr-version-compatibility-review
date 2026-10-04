"""Bounded evidence investigation. Retrieved candidates never change static risks."""
from __future__ import annotations
import copy
import hashlib
import inspect
import json

VERSIONS=('v2.9.1','v3.0.0')


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
                       'application':finding['application'],'required_tokens':{}})
    for gap in report.get('gaps',[]):
        if len(checks)>=12:break
        key=gap['code']
        if key in seen:continue
        seen.add(key)
        checks.append({'check_id':f'check-{len(checks)+1}','query':'PaddleOCR '+gap['detail'][:400],
                       'versions':list(VERSIONS),'gap_code':key,'required_tokens':{}})
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
    stop='COMPLETED'; queue=list(checks); seen_queries=set(); processed=0
    def remaining():return max(0,deadline_seconds-prior_elapsed-(clock()-started))
    for round_no in range(1,4):
        batch=queue[:4];queue=queue[4:]
        if not batch:break
        for batch_index,check in enumerate(batch):
            processed+=1; resolved_versions=[]
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
                good=[]
                for hit in cached.get('results',[])[:20]:
                    try:
                        if validate_evidence is None:raise ValueError('source validator not configured')
                        if hit.get('version')!=version:raise ValueError('wrong version')
                        original=validate_evidence(hit,remaining())
                        if not isinstance(original,dict) or original.get('chunk_id')!=hit.get('chunk_id'):raise ValueError('identity')
                        if any(t not in original.get('content','') for t in check.get('required_tokens',{}).get(version,[])):continue
                        checked[original['chunk_id']]=original;good.append(original['chunk_id'])
                    except Exception:
                        trace.append({'check_id':check['check_id'],'version':version,'status':'EVIDENCE_VALIDATION_REQUIRED',
                                      'query':check['query'],'evidence_id':hit.get('chunk_id')})
                if good:resolved_versions.append(version)
                trace.append({'check_id':check['check_id'],'version':version,'query':check['query'],
                              'status':'SOURCE_VERIFIED_CANDIDATES' if good else 'NO_VERIFIED_EVIDENCE',
                              'evidence_ids':good,'cached':key in (resume or {}).get('completed',{})})
            missing=[v for v in check['versions'] if v not in resolved_versions]
            if missing:unresolved.append({'check_id':check['check_id'],'versions':missing,
                                         'reason':'required_source_not_verified','query':check['query']})
            if stop in ('CANCELLED','DEADLINE_EXCEEDED','DEADLINE_NOT_ENFORCEABLE','SEARCH_BUDGET_EXHAUSTED'):
                queue=batch[batch_index+1:]+queue
                break
        if stop in ('CANCELLED','DEADLINE_EXCEEDED','DEADLINE_NOT_ENFORCEABLE','SEARCH_BUDGET_EXHAUSTED'):break
        if planner and unresolved and remaining()>0 and state['planner_calls']<3 and processed+len(queue)<12:
            try:
                if not _bounded_method(planner):raise ValueError('unbounded planner')
                state['planner_calls']+=1
                proposal=planner(unresolved=copy.deepcopy(unresolved),request_timeout=remaining())
                parents={c['check_id']:c for c in checks}
                for row in proposal[:min(4,12-processed-len(queue))]:
                    parent=parents.get(row.get('parent_check_id'))
                    if parent is None or not row.get('versions') or not set(row['versions']).issubset(parent['versions']):continue
                    query=row.get('query')
                    if not isinstance(query,str) or not 0<len(query)<=4000:continue
                    queue.append({**parent,'check_id':f"{parent['check_id']}-r{round_no}",'query':query,'versions':row['versions']})
                trace.append({'status':'PLANNER_COMPLETED','round':round_no,'proposed_count':len(proposal)})
            except Exception:
                trace.append({'status':'PLANNER_FAILED','round':round_no})
    for c in queue:unresolved.append({'check_id':c['check_id'],'versions':c['versions'],'reason':'ROUND_OR_ITEM_BUDGET'})
    state['elapsed_seconds']=min(240,max(0,prior_elapsed+clock()-started))
    if unresolved and stop=='COMPLETED':stop='NO_NEW_EVIDENCE'
    return {'schema_version':1,'stop_reason':stop,'trace':trace,'checked_evidence':list(checked.values()),
            'unresolved':unresolved,'search_calls':state['search_calls'],'planner_calls':state['planner_calls'],
            'elapsed_seconds':state['elapsed_seconds'],'resume':state,
            'assessment':'source_verified_candidates_not_confirmed_business_impact'}
