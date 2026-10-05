"""Scope-first retrieval over evidence windows, with explicit neural fallbacks."""
from __future__ import annotations
import math
import hashlib
import json
import os
import threading
from collections import Counter,defaultdict
import numpy as np
from src.public_knowledge import tokens
from src.paddleocr_retrieval_views import build_views, validate_views, materialize_evidence

STRATEGIES=('contextual_bm25','contextual_semantic','contextual_rrf',
            'contextual_rerank','contextual_rrf_rerank','contextual_llm_rerank')
_CACHE={}
_LOCK=threading.Lock()


def configured_evidence_search(index,strategy='contextual_bm25',*,use_context=True):
    base=getattr(index,'base_index',index)
    if base.manifest.get('workspace_id')!='paddleocr':raise ValueError('wrong evidence workspace')
    fingerprint=hashlib.sha256(''.join(c['chunk_id']+c['content'] for c in base.chunks).encode()).hexdigest()
    from src.paddleocr_quality import asset_identity
    key=(fingerprint,strategy,use_context,asset_identity(os.environ.get('PADDLEOCR_EMBEDDING_PATH','')),asset_identity(os.environ.get('PADDLEOCR_RERANKER_PATH','')))
    with _LOCK:
        if key in _CACHE:return _CACHE[key]
        encoder=scorer=count=None
        if strategy not in ('contextual_bm25','contextual_llm_rerank'):
            from src.paddleocr_quality import configured_quality_retriever, ModelUnavailable
            try:
                model=configured_quality_retriever(base)
                encoder,scorer,count=model.encoder,model.scorer,getattr(model,'token_count',None)
            except ModelUnavailable:
                pass
        views=build_views(base.chunks,token_count=count or len,max_tokens=384)
        if not use_context:
            views=[{**v,'context_char_end':0,'retrieval_text':v['content']} for v in views]
        search=EvidenceSearch(base,views,encoder=encoder,scorer=scorer,token_count=count)
        search.model_identity={'embedding':key[-2],'reranker':key[-1],'attention':'eager'}
        _CACHE[key]=search
        return search


class EvidenceSearch:
    def __init__(self,index,views,*,encoder=None,scorer=None,token_count=None):
        self.index=getattr(index,'base_index',index)
        validate_views(self.index.chunks,views)
        self.views=views; self.parents={c['chunk_id']:c for c in self.index.chunks}
        self.encoder,self.scorer,self.token_count=encoder,scorer,token_count
        self.vectors=None
        self.tf=[Counter(tokens(v['retrieval_text'])) for v in views]
        self.df=Counter(t for row in self.tf for t in row)
        self.lengths=[sum(c.values()) for c in self.tf]
        self.avg=sum(self.lengths)/max(1,len(self.lengths))

    def _lexical(self,query,eligible):
        qt=Counter(tokens(query)); n=len(self.views); rows=[]
        for i in eligible:
            tf=self.tf[i]; score=0.0
            for t,count in qt.items():
                if t not in tf:continue
                idf=math.log(1+(n-self.df[t]+.5)/(self.df[t]+.5))
                score+=idf*tf[t]*2.5/(tf[t]+1.5*(.25+.75*self.lengths[i]/max(1,self.avg)))*count
            if score>0:rows.append((i,score))
        return sorted(rows,key=lambda x:(-x[1],x[0]))

    def search(self,plan:dict,*,top_k:int=5,candidate_budget:int=80,strategy:str='contextual_bm25',ranker=None)->dict:
        if type(top_k) is not int or not 1<=top_k<=20 or candidate_budget not in (40,80,120) or strategy not in STRATEGIES:
            raise ValueError('invalid evidence search options')
        if not isinstance(plan.get('versions'),list) or not plan['versions']:
            raise ValueError('missing version scope')
        eligible=[i for i,v in enumerate(self.views) if v['version'] in plan['versions']
                  and self.parents[v['parent_chunk_id']].get('language','zh')==plan.get('language','zh')
                  and self.parents[v['parent_chunk_id']].get('namespace','project_primary')==plan.get('namespace','project_primary')]
        modules={r['module'] for r in plan.get('requirements',[]) if r.get('module')}
        # Apply explicit pipeline scope before candidate truncation. Identical
        # result fields across table/structure/OCR are not interchangeable.
        if modules:
            api_names={'ocr':'PaddleOCR', 'structure':'PPStructureV3',
                       'doc_preprocessor':'DocPreprocessor', 'table_recognition_v2':'TableRecognitionPipelineV2'}
            eligible=[i for i in eligible if self.views[i]['module'] in modules|{'shared'}
                      or (self.views[i]['module']=='general' and any(
                          api_names.get(m,'\0') in self.views[i]['retrieval_text'] for m in modules))]
        diag={'requested_strategy':strategy,'actual_strategy':strategy,'candidate_budget':candidate_budget,
              'eligible_windows':len(eligible),'rerank_calls':0,'external_model_calls':0,
              'fallback_reason':None,'window_failures':[]}
        diag['model_identity']=getattr(self,'model_identity',{'embedding':'NOT_CONFIGURED','reranker':'NOT_CONFIGURED'})
        if plan.get('status')!='READY':
            return {'results':[],'requirements':plan.get('requirements',[]),'diagnostics':{**diag,'status':'NEEDS_CLARIFICATION'}}
        queries=list(dict.fromkeys([plan['original_query']]+[r['query'] for r in plan['requirements']]+plan.get('proposed_queries',[])))[:9]
        pools=[self._lexical(q,eligible)[:candidate_budget] for q in queries]
        weights=defaultdict(float)
        for pool in pools:
            for rank,(i,_) in enumerate(pool,1):weights[i]+=1/(60+rank)
        lexical_order=sorted(weights,key=lambda i:(-weights[i],i))
        order=lexical_order
        if strategy in ('contextual_semantic','contextual_rrf','contextual_rrf_rerank'):
            try:
                if self.encoder is None:raise ValueError('SEMANTIC_MODEL_NOT_CONFIGURED')
                if self.vectors is None:self.vectors=np.asarray(self.encoder([v['retrieval_text'] for v in self.views]),dtype=np.float32)
                q=np.asarray(self.encoder(['为这个句子生成表示以用于检索相关文章：'+plan['original_query']]),dtype=np.float32)
                if (q.ndim!=2 or q.shape[0]!=1 or self.vectors.shape!=(len(self.views),q.shape[1])
                        or not np.isfinite(q).all() or not np.isfinite(self.vectors).all()):raise ValueError('SEMANTIC_INVALID_OUTPUT')
                scores=self.vectors[eligible]@q[0]
                semantic=sorted(range(len(eligible)),key=lambda j:(-float(scores[j]),eligible[j]))[:candidate_budget]
                semantic=[eligible[j] for j in semantic]
                if strategy=='contextual_semantic':order=semantic; weights={i:float(scores[eligible.index(i)]) for i in order}
                else:
                    fused=defaultdict(float)
                    for pool in (lexical_order,semantic):
                        for rank,i in enumerate(pool,1):fused[i]+=1/(60+rank)
                    weights=fused; order=sorted(fused,key=lambda i:(-fused[i],i))
            except (ValueError,RuntimeError,OSError) as exc:
                diag.update(actual_strategy='contextual_bm25',fallback_reason=str(exc)); order=lexical_order
        order=order[:candidate_budget]; diag['candidate_windows']=len(order)
        candidate_evidence=materialize_evidence(self.index.chunks,[self.views[i] for i in order])
        selected_views=[]
        if strategy == 'contextual_llm_rerank' and order:
            try:
                if not callable(ranker): raise ValueError('LLM_RANKER_NOT_CONFIGURED')
                views = [self.views[i] for i in order[:40]]
                rows = [{'chunk_id': v['view_id'], 'excerpt': v['retrieval_text'],
                         'title': self.parents[v['parent_chunk_id']].get('document_title',''),
                         'path': self.parents[v['parent_chunk_id']].get('document_path',''),
                         'version': v['version']} for v in views]
                diag['external_model_calls'] = 1
                ids, details = ranker(task=plan['original_query'], candidates=rows)
                by_id = {v['view_id']: v for v in views}
                if not isinstance(ids,list) or len(ids)!=len(by_id) or set(ids)!=set(by_id):
                    raise ValueError('LLM_RANK_INVALID_PERMUTATION')
                selected_views = [{**by_id[cid], 'rank_score': float(len(ids)-i)} for i,cid in enumerate(ids)]
                diag.update(rerank_calls=1, scored_windows=len(ids), score_type='model_order_not_probability',
                            llm_ranking=details)
            except Exception as exc:
                diag.update(actual_strategy='contextual_bm25', fallback_reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__)
        elif 'rerank' in strategy and order:
            try:
                if self.scorer is None:raise ValueError('RERANK_MODEL_NOT_CONFIGURED')
                # Query+document tokenizer budget, never silent truncation.
                count=self.token_count or len
                query_tokens=count(plan['original_query']); budget=min(384,480-query_tokens)
                if budget<16:raise ValueError('QUERY_EXCEEDS_RERANK_WINDOW_BUDGET')
                candidates=[]
                for i in order:
                    view=self.views[i]
                    if count(view['retrieval_text'])>budget:
                        parent=self.parents[view['parent_chunk_id']]
                        windows=build_views([parent],token_count=count,max_tokens=budget)
                        candidates.extend(w for w in windows if w['char_start']<view['char_end'] and w['char_end']>view['char_start'])
                    else:candidates.append(view)
                # Extra subwindows cannot silently expand the total scoring budget.
                unique={v['view_id']:v for v in candidates}
                candidates=list(unique.values())
                if len(candidates)>candidate_budget:
                    # Pick split windows by query overlap, rather than dropping
                    # the tail merely because it appears later in a long table.
                    qt=Counter(tokens(plan['original_query']))
                    candidates.sort(key=lambda v:-sum(qt[t] for t in tokens(v['retrieval_text']) if t in qt))
                    diag['subwindow_budget_excluded']=len(candidates)-candidate_budget
                    candidates=candidates[:candidate_budget]
                scores=np.asarray(self.scorer([(plan['original_query'],v['retrieval_text']) for v in candidates])).reshape(-1)
                if len(scores)!=len(candidates) or not np.isfinite(scores).all():raise ValueError('RERANK_INVALID_OUTPUT')
                ranked=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),i))
                selected_views=[{**candidates[i],'rank_score':float(scores[i])} for i in ranked]
                diag.update(rerank_calls=1,scored_windows=len(candidates),score_type='cross_encoder_logit')
                if diag['actual_strategy']=='contextual_bm25':diag['actual_strategy']='contextual_rerank'
            except (ValueError,RuntimeError,OSError) as exc:
                diag.update(actual_strategy='contextual_bm25',fallback_reason=str(exc)); order=lexical_order[:candidate_budget]
        if not selected_views:
            selected_views=[{**self.views[i],'rank_score':float(weights.get(i,0))} for i in order]
        # Reserve a relevant source for each explicit requirement before filling
        # the global Top-K. A high score for one clause must not crowd out another.
        reserved=[]; coverage={}
        for req in plan['requirements']:
            lexical_ids={self.views[i]['parent_chunk_id'] for i,_ in self._lexical(req['query'],eligible)}
            matching=[v for v in selected_views if v['parent_chunk_id'] in lexical_ids
                      and (not req.get('version') or v['version']==req['version'])
                      and (not req.get('module') or v['module'] in (req['module'],'shared'))
                      and all(s in v['retrieval_text'] for s in req.get('symbols',[]) if s!='PaddleOCR')]
            best=next((v for v in matching if v['parent_chunk_id'] not in {r['parent_chunk_id'] for r in reserved}), None)
            if best is None and matching: best=matching[0]
            coverage[req['id']]=best['parent_chunk_id'] if best else None
            if best and best not in reserved and len(reserved)<top_k: reserved.append(best)
        selected_views=reserved+[v for v in selected_views if v not in reserved]
        diag['requirement_candidates']=coverage
        parents=[]; chosen=[]
        for view in selected_views:
            pid=view['parent_chunk_id']
            if pid not in parents:
                if len(parents)>=top_k:continue
                parents.append(pid)
            # At most 4 relevant windows per source block; exclude irrelevant additional tails.
            if sum(v['parent_chunk_id']==pid for v in chosen)<4:chosen.append(view)
        hits=materialize_evidence(self.index.chunks,chosen)
        for rank,hit in enumerate(hits,1):
            hit.update(rank=rank,retrieval_policy=diag['actual_strategy'],
                       retrieval_score=max(v['rank_score'] for v in chosen if v['parent_chunk_id']==hit['chunk_id']),
                       score_type=diag.get('score_type','window_rrf_rank_score'))
        diag.update(final_evidence_count=len(hits),selected_windows=len(chosen),status='OK' if hits else 'NO_CANDIDATES')
        return {'results':hits,'requirements':plan['requirements'],'diagnostics':diag,'candidate_evidence':candidate_evidence}
