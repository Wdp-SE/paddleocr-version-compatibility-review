"""Conservative model assessment of claim entailment, not a truth probability."""
from __future__ import annotations
import json
from src.paddleocr_retrieval_views import generation_evidence_text as evidence_text
from src.paddleocr_retrieval_views import module_for


def check_claim_support(claims, evidence, judge, *, question=None):
    failure={'status':'CHECK_FAILED','claims':[],'rejected':[],'diagnostics':{},
             'failure_type':'InvalidCheckerResponse',
             'assessment_type':'model_assessment_not_human_validation'}
    payload=[]
    try:
        by_id={row['chunk_id']:row for row in evidence}
        if not isinstance(claims,list) or not 1<=len(claims)<=5:
            return failure
        for i,claim in enumerate(claims):
            ids=claim['evidence_ids']
            if not ids or any(cid not in by_id for cid in ids):return failure
            payload.append({'claim_index':i,'claim':claim['text'],
                            'evidence':[{'id':cid,'version':by_id[cid].get('version'),
                                         'module':by_id[cid].get('module',module_for(by_id[cid])),
                                         'document_path':by_id[cid].get('document_path'),
                                         'content':evidence_text(by_id[cid])} for cid in ids]})
            if question is not None:
                if not isinstance(question,str) or not 0<len(question)<=4000:return failure
                payload[-1]['question']=question
        if len(json.dumps(payload,ensure_ascii=False))>60000:
            return {**failure,'failure_type':'EvidenceBudgetExceeded'}
        result,diagnostics=judge(payload)
        failure={**failure,'diagnostics':diagnostics}
        rows=result['verdicts']
        if not isinstance(rows,list) or len(rows)!=len(claims):return failure
        seen=set()
        for row in rows:
            i=row['claim_index']
            if (type(i) is not int or not 0<=i<len(claims) or i in seen
                    or type(row['supported']) is not bool
                    or not isinstance(row['reason'],str) or not 0<len(row['reason'])<=800):
                return failure
            seen.add(i)
        supported={row['claim_index'] for row in rows if row['supported']}
        rejected=[row for row in rows if not row['supported']]
        return {'status':'SUPPORTED' if len(supported)==len(claims) else 'PARTIAL_SUPPORTED' if supported else 'UNSUPPORTED',
                'claims':[claim for i,claim in enumerate(claims) if i in supported],
                'rejected':rejected,'diagnostics':diagnostics,
                'assessment_type':'model_assessment_not_human_validation'}
    except Exception as exc:
        # A checker failure must not promote an unchecked answer. Preserve safe category only.
        return {**failure,'failure_type':type(exc).__name__}
