"""Scope-first retrieval over evidence windows, with explicit neural fallbacks."""
from __future__ import annotations
import math
import hashlib
import json
import os
import re
import threading
from collections import Counter,defaultdict
import numpy as np
from src.paddleocr_evidence_units import (retrieval_tokens as tokens, is_layout_only,
                                        parameter_definition_priority, query_definition_symbols,
                                        task_definition_priority)
from src.paddleocr_evidence_units import behavior_context_priority
from src.paddleocr_retrieval_views import build_views, build_structure_views, validate_views, materialize_evidence

STRATEGIES=('contextual_bm25','contextual_semantic','contextual_rrf',
            'contextual_rerank','contextual_rrf_rerank','contextual_llm_rerank')
_CACHE={}
_LOCK=threading.Lock()


def configured_evidence_search(index,strategy='contextual_bm25',*,use_context=True,window_budget=384,structure_windows=False):
    if type(window_budget) is not int or window_budget not in (256,384):
        raise ValueError('unsupported evidence window budget')
    base=getattr(index,'base_index',index)
    if base.manifest.get('workspace_id')!='paddleocr':raise ValueError('wrong evidence workspace')
    fingerprint=hashlib.sha256(''.join(c['chunk_id']+c['content'] for c in base.chunks).encode()).hexdigest()
    from src.paddleocr_quality import asset_identity
    key=(fingerprint,strategy,use_context,window_budget,structure_windows,'definition_units_v1',asset_identity(os.environ.get('PADDLEOCR_EMBEDDING_PATH','')),asset_identity(os.environ.get('PADDLEOCR_RERANKER_PATH','')))
    with _LOCK:
        if key in _CACHE:return _CACHE[key]
        encoder=scorer=count=None
        if strategy not in ('contextual_bm25','contextual_llm_rerank'):
            from src.paddleocr_quality import configured_quality_retriever, ModelUnavailable
            try:
                model=configured_quality_retriever(base,
                    load_embedding=strategy in ('contextual_semantic','contextual_rrf','contextual_rrf_rerank'),
                    load_reranker='rerank' in strategy,build_parent_vectors=False)
                encoder,scorer,count=model.encoder,model.scorer,getattr(model,'token_count',None)
            except ModelUnavailable:
                pass
        # All channels share identical retrieval units. Tokenizer counts are
        # used only to bound cross-encoder input, not to change the comparison.
        views=(build_structure_views(base.chunks,max_chars=window_budget) if structure_windows
               else build_views(base.chunks,token_count=len,max_tokens=window_budget))
        if not use_context:
            views=[{**v,'context_char_end':0,'retrieval_text':v['content']} for v in views]
        search=EvidenceSearch(base,views,encoder=encoder,scorer=scorer,token_count=count)
        search.model_identity={'embedding':key[-2],'reranker':key[-1],'attention':'eager','cpu_kernel':'native_serial_v2'}
        search.window_budget = window_budget
        search.window_unit = 'structured_characters' if structure_windows else 'characters'
        _CACHE[key]=search
        return search


class EvidenceSearch:
    def __init__(self,index,views,*,encoder=None,scorer=None,token_count=None):
        self.index=getattr(index,'base_index',index)
        validate_views(self.index.chunks,views)
        self.parents={c['chunk_id']:c for c in self.index.chunks}
        from src.paddleocr_applicability import example_applicability
        self.applicability=example_applicability(self.index.chunks)
        views=[v for v in views if not is_layout_only(v['content'],self.parents[v['parent_chunk_id']].get('document_path',''))]
        self.views=views
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
        symbols=query_definition_symbols(query)
        return sorted(rows,key=lambda x:(-parameter_definition_priority(self.views[x[0]]['content'],symbols),-x[1],x[0]))

    def search(self,plan:dict,*,top_k:int=5,candidate_budget:int=80,strategy:str='contextual_bm25',ranker=None,
               max_windows_per_source:int=4)->dict:
        if type(top_k) is not int or not 1<=top_k<=20 or candidate_budget not in (40,80,120) or strategy not in STRATEGIES:
            raise ValueError('invalid evidence search options')
        if type(max_windows_per_source) is not int or not 1 <= max_windows_per_source <= 8:
            raise ValueError('invalid context window budget')
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
        diag['context_window_limit_per_source'] = max_windows_per_source
        diag['model_identity']=getattr(self,'model_identity',{'embedding':'NOT_CONFIGURED','reranker':'NOT_CONFIGURED'})
        diag['window_budget']=getattr(self,'window_budget',None)
        diag['window_unit']=getattr(self,'window_unit','caller_supplied')
        if plan.get('status')!='READY':
            return {'results':[],'requirements':plan.get('requirements',[]),'diagnostics':{**diag,'status':'NEEDS_CLARIFICATION'}}
        intent=plan.get('retrieval_intent',{})
        symbols=list(dict.fromkeys(s for r in plan['requirements'] for s in r.get('symbols',[]) if s!='PaddleOCR'))
        queries=list(dict.fromkeys(plan.get('proposed_queries',[])+[plan['original_query']]+[r.get('retrieval_query',r['query']) for r in plan['requirements']]
                                  +plan.get('proposed_queries',[])+intent.get('aliases',[])+symbols))[:9]
        pools=[self._lexical(q,eligible)[:candidate_budget] for q in queries]
        diag['repair_requirement_ids'] = plan.get('repair_requirement_ids', [])
        diag['query_count'] = len(queries)
        weights=defaultdict(float)
        for pool in pools:
            for rank,(i,_) in enumerate(pool,1):weights[i]+=1/(60+rank)
        lexical_order=sorted(weights,key=lambda i:(-weights[i],i))
        diag.update(lexical_candidates=min(len(lexical_order),candidate_budget),semantic_candidates=0)
        # Prefer the artifact requested by the user, without excluding other
        # source roles or inventing facts. Only already relevant windows qualify.
        def field_definition_priority(view):
            parent=self.parents[view['parent_chunk_id']]
            if not parent.get('document_path','').endswith('.md'):return 0
            return sum(bool(re.search(r'(?m)^\s*[-*]\s*`?'+name+r'`?\s*[:：]',parent['content']))
                       for name in ('rec_texts','rec_scores'))

        def role_priority(i):
            parent=self.parents[self.views[i]['parent_chunk_id']]
            path=parent.get('document_path','').lower()
            text=self.views[i]['retrieval_text']
            if intent.get('kind')=='parameter_behavior':
                return behavior_context_priority(self.views[i]['content'],symbols)
            if intent.get('kind')=='parameter_lookup':
                return task_definition_priority(self.views[i]['content'],intent)
            if intent.get('kind')=='result_usage':
                methods=intent.get('methods',[])
                calls=sum(bool(re.search(r'\.'+re.escape(method)+r'\s*\(',text)) for method in methods)
                definition=sum(method in text and 'save_path' in text for method in methods if method.startswith('save_to_'))
                return calls*2 + definition*3 + int(bool(re.search(r'for\s+\w+\s+in\s+',text)))
            if intent.get('kind')=='result_fields':
                fields=['rec_texts','rec_scores']
                count=sum(field in text for field in fields)
                definition=bool(re.search(r'<td\b|\|.*\|', self.views[i]['content'],re.I))
                return count + 2*int(definition and count>0) + field_definition_priority(self.views[i])
            if intent.get('kind') == 'parameter_migration' and path.endswith('.py'):
                body = self.views[i]['content']
                if any(t in body for t in ('kwargs.items()', 'base_params[name]', 'super().__init__(**base_params)')):
                    return 4
                if any(t in body for t in ('parse_common_args(', 'unknown_names', 'default_vals =')):
                    return 3
                if '_DEPRECATED_PARAM_NAME_MAPPING =' in body:
                    return 2
                return 0
            if intent.get('kind')=='result_migration':
                # An interface comparison needs callable behavior as well as
                # printed fields; serialized examples alone do not define types.
                if 'return list(' in text and 'self.predict_iter(' in text:return 4
                if 'ocr_res.append' in text or ('def ocr(' in text and 'self.predict' in text):return 3
                return 2 if 'rec_texts' in text or 'rec_scores' in text or 'line[1]' in text else 0
            if intent.get('kind')=='test_contract':
                return 2 if path.startswith('tests/') and 'assert ' in text else 1 if path.startswith('tests/') else 0
            if intent.get('kind')=='api_usage':
                is_api=(path.endswith('.py') or path.endswith('/whl.md'))
                if intent.get('method_lookup'):
                    if 'def ocr(' in text and 'self.predict(' in text:
                        return 4
                    if '.predict(' in text and 'PaddleOCR(' in text:
                        return 3
                    if 'def predict(' in text:
                        return 2
                if not is_api:return 0
                if 'ocr det rec False' in intent.get('aliases', []):
                    return 2 if 'det=False' in text or 'det = False' in text else 0
                if re.search(r'配合|\bocr\s*\(', plan['original_query'], re.I):
                    return 3 if 'PaddleOCR(' in text and '.ocr(' in text and all(s in text for s in symbols) else 0
                # A generic parameter question does not justify promoting an
                # unrelated constructor example over its exact parameter row.
                return 0
            return 0
        lexical_order.sort(key=lambda i:(-role_priority(i),-weights[i],i))
        diag['source_intent']=intent.get('kind','knowledge')
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
                diag['semantic_candidates']=len(semantic)
                if strategy=='contextual_semantic':order=semantic; weights={i:float(scores[eligible.index(i)]) for i in order}
                else:
                    fused=defaultdict(float)
                    for pool in (lexical_order,semantic):
                        for rank,i in enumerate(pool,1):fused[i]+=1/(60+rank)
                    weights=fused; order=sorted(fused,key=lambda i:(-fused[i],i))
            except (ValueError,RuntimeError,OSError) as exc:
                diag.update(actual_strategy='contextual_bm25',fallback_reason=str(exc)); order=lexical_order
        # Reserve one in-scope witness for each explicit fact dimension BEFORE
        # the fused global cutoff. Otherwise a multi-part question can retrieve
        # many variants of one clause and erase another clause's whole pool.
        facet_seeds=[]
        for req in plan['requirements']:
            dimension=req.get('fact_dimension')
            if not dimension:continue
            local=[i for i in eligible if (not req.get('version') or self.views[i]['version']==req['version'])
                   and (not req.get('module') or self.views[i]['module'] in (req['module'],'shared'))]
            ranked=self._lexical(req.get('retrieval_query',req['query']),local)
            def facet_priority(i):
                view=self.views[i];body=view['content']
                own=view['module']==req.get('module')
                if dimension=='parameters':return (int('def __init__(' in body),own)
                if dimension=='call':
                    preferred='predict' if view['version']=='v3.0.0' else 'ocr'
                    return (2*int('def '+preferred+'(' in body)+int(bool(re.search(r'def (?:ocr|predict)\(',body))),own)
                score=(3*int('ocr_res.append' in body) if view['version']=='v2.9.1' else
                       3*field_definition_priority(view)+sum(s in body for s in ('rec_texts','rec_scores')))
                return (score,own)
            ranked.sort(key=lambda pair:(*[-int(x) for x in facet_priority(pair[0])],-pair[1]))
            if ranked and facet_priority(ranked[0][0])[0]>0:facet_seeds.append(ranked[0][0])
        order=list(dict.fromkeys(facet_seeds+order))[:candidate_budget]
        diag['facet_candidate_reservations']=len(set(facet_seeds))
        diag['candidate_windows']=len(order)
        candidate_evidence=materialize_evidence(self.index.chunks,[self.views[i] for i in order],applicability=self.applicability)
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
            ranked=self._lexical(req.get('retrieval_query',req['query']),eligible)
            lexical_ids={self.views[i]['parent_chunk_id'] for i,_ in ranked}
            lexical_rank={self.views[i]['view_id']:rank for rank,(i,_) in enumerate(ranked)}
            from src.paddleocr_applicability import version_applies
            matching=[v for v in selected_views if v['parent_chunk_id'] in lexical_ids
                      and (not req.get('version') or version_applies(
                          {**self.parents[v['parent_chunk_id']],
                           'applicability':self.applicability.get(v['parent_chunk_id'])},req['version']))
                      and (not req.get('module') or v['module'] in (req['module'],'shared'))
                      and all(s in v['retrieval_text'] for s in req.get('symbols',[]) if s!='PaddleOCR')]
            from src.paddleocr_context_selection import definition_score
            def dimension_score(view):
                text=view['content'];dimension=req.get('fact_dimension')
                if dimension=='call':
                    preferred='predict' if view['version']=='v3.0.0' else 'ocr'
                    return 3*int('def '+preferred+'(' in text)+int(bool(re.search(r'def (?:ocr|predict)\(|\.ocr\(|\.predict\(',text)))
                if dimension=='result':
                    if view['version']=='v2.9.1':return 3*int('ocr_res.append' in text)+int('line[1][0]' in text)
                    return 3*field_definition_priority(view)+sum(t in text for t in ('rec_texts','rec_scores'))
                if dimension=='parameters':
                    return 3*int('def __init__(' in text)+int(any(t in text for t in ('_DEPRECATED_PARAM_NAME_MAPPING','参数','默认值')))
                return 0
            matching.sort(key=lambda v:(-dimension_score(v),int(bool(req.get('module')) and v['module']!=req['module']),
                                         *[-n for n in definition_score(v,req.get('symbols',[]))],
                                         lexical_rank.get(v['view_id'],candidate_budget)))
            # One source may cover multiple clauses; do not spend Top-K on a
            # weaker parent merely because the best witness is already used.
            best=matching[0] if matching else None
            coverage[req['id']]=best['parent_chunk_id'] if best else None
            used_parents={v['parent_chunk_id'] for v in reserved}
            if best and best not in reserved and (best['parent_chunk_id'] in used_parents or len(used_parents)<top_k):
                reserved.append(best)
        requirement_witnesses=list(reserved)
        if intent.get('kind') == 'api_usage' and re.search(r'配合|\bocr\s*\(', plan['original_query'], re.I):
            witness = next((v for v in selected_views if 'PaddleOCR(' in v['content'] and '.ocr(' in v['content']
                            and all(s in v['content'] for s in symbols)), None)
            if witness:
                reserved = [witness] + [v for v in reserved if v != witness]
                reserved = reserved[:top_k]
        if intent.get('kind')=='result_migration':
            # Callable contracts are required evidence, independent of rerank preference.
            # Only use retrieved, in-scope code; never synthesize an API contract.
            contracts=[]
            for version in plan['versions']:
                contract=next((v for v in selected_views if v['version']==version
                               and self.parents[v['parent_chunk_id']].get('document_path','').endswith('.py')
                               and 'return list(' in v['retrieval_text'] and 'self.predict_iter(' in v['retrieval_text']),None)
                contract=contract or next((v for v in selected_views if v['version']==version
                               and self.parents[v['parent_chunk_id']].get('document_path','').endswith('.py')
                               and ('ocr_res.append' in v['retrieval_text'] or
                                    ('def ocr(' in v['retrieval_text'] and 'self.predict' in v['retrieval_text']))),None)
                if contract and contract not in contracts:contracts.append(contract)
            for version in plan['versions']:
                access=next((v for v in selected_views if v['version']==version
                             and self.parents[v['parent_chunk_id']].get('document_path','').startswith('tests/')
                             and 'assert ' in v['retrieval_text']
                             and any(t in v['retrieval_text'] for t in ('rec_texts','rec_scores'))),None)
                if access and access not in contracts:contracts.append(access)
            reserved=(contracts+reserved)[:top_k]
        if intent.get('kind')=='result_migration':
            # Keep both the callable behavior and a field contract in an upgrade
            # evidence pack; repeated old usage examples must not crowd it out.
            for version in plan['versions']:
                fields=[v for v in selected_views if v['version']==version
                            and any(t in v['retrieval_text'] for t in ('rec_texts','rec_scores'))
                            and all(t in self.parents[v['parent_chunk_id']]['content'] for t in ('rec_texts','rec_scores'))]
                # Prefer the field description to a serialized sample or test
                # assertion when the task asks what the result means.
                fields.sort(key=lambda v:(
                    -field_definition_priority(v),
                    -int(self.parents[v['parent_chunk_id']].get('document_path','').endswith('.md')
                         and not self.parents[v['parent_chunk_id']]['content'].lstrip().startswith('```')),
                    -sum(t in v['content'] for t in ('rec_texts','rec_scores'))))
                field=fields[0] if fields else None
                if field and field not in reserved:
                    # Field schema is a distinct contract obligation. Reserve it
                    # before repeated usage windows consume the evidence budget.
                    reserved = (contracts + [field] + [v for v in reserved if v not in contracts])[:top_k]
            # The outer [box, recognition] shape alone cannot establish the
            # meaning of a nested index. Keep a retrieved usage witness too,
            # rather than asking the generator to infer text/score positions.
            if top_k >= 5:
                old_usage = next((v for v in selected_views
                                  if v['version'] == 'v2.9.1'
                                  and all(symbol in v['retrieval_text'].replace(' ', '')
                                          for symbol in ('line[1][0]', 'line[1][1]'))), None)
                if old_usage and old_usage not in reserved:
                    reserved = reserved[:top_k-1] + [old_usage]
        if intent.get('kind') == 'parameter_migration':
            from src.paddleocr_context_selection import complete_parameter_context, parameter_contract_windows
            selected_views = complete_parameter_context(selected_views, self.parents)
            anchors = parameter_contract_windows(selected_views)
            anchor_ids = {v['view_id'] for v in anchors}
            selected_views = [{**v, 'contract_anchor': v['view_id'] in anchor_ids} for v in selected_views]
            anchors = [v for v in selected_views if v['contract_anchor']]
            # Keep version obligations and then the bounded parameter chain;
            # parameter tables alone cannot prove a forwarded keyword is rejected.
            # Multiple clauses can reserve several old-version parents. Keep
            # one baseline first so the target's mapping, forwarding, base and
            # validation parents can all fit in the default five-source budget.
            baseline = [v for v in reserved if v['version'] == plan['versions'][0]][:1]
            reserved = baseline + anchors + [v for v in reserved if v not in baseline]
        if intent.get('kind')=='parameter_behavior':
            from src.paddleocr_context_selection import complete_parameter_context
            selected_views=complete_parameter_context(selected_views,self.parents)
            branches=[{**v,'contract_anchor':True} for v in selected_views
                      if behavior_context_priority(v['content'],symbols)]
            reserved=branches+reserved
            # A boolean guard alone cannot show how the per-call flag arrives.
            forwarding=[{**v,'contract_anchor':True,'context_priority':2} for v in selected_views
                        if re.search(r'self\.__call__\([^\n]*\bcls\b',v['content'])
                        and 'cls' in intent.get('symbols',[])]
            reserved=forwarding+reserved
        if intent.get('kind') == 'parameter_lookup':
            # Preserve definition rows from different retrieved sources. A new
            # implementation file must not hide a conflicting guide/default.
            definitions, seen_parents = [], set()
            for view in selected_views:
                if (task_definition_priority(view['content'], intent) and
                        view['parent_chunk_id'] not in seen_parents):
                    definitions.append({**view, 'contract_anchor': True})
                    seen_parents.add(view['parent_chunk_id'])
            reserved = definitions[:min(3, top_k)] + reserved
        if intent.get('kind') == 'parameter_lookup' and any('batch' in s.lower() for s in symbols):
            batch_anchors = []
            for token in ('range(0, img_num, batch_num)', 'text_recognizer(img_crop_list)'):
                match = next((v for v in selected_views if token in v['content']), None)
                if match:
                    batch_anchors.append({**match, 'contract_anchor': True})
            reserved = reserved + batch_anchors
        if (any(r.get('fact_dimension')=='parameters' for r in plan['requirements'])
                and any(r.get('fact_dimension') in ('call','result') for r in plan['requirements'])):
            from src.paddleocr_context_selection import initializer_context
            initializers=initializer_context(selected_views,self.parents)
            constructors=[]
            for version in plan['versions']:
                candidates=[v for v in initializers if v['version']==version]
                candidates.sort(key=lambda v:(v.get('module')!='ocr',
                    self.parents[v['parent_chunk_id']].get('document_path') not in ('paddleocr.py','paddleocr/_pipelines/ocr.py')))
                if candidates:constructors.append(candidates[0])
            # Explicit dimensions outrank optional illustrative tests. Count
            # source parents, not windows, so one shared witness can serve two
            # obligations without hiding the final result-schema requirement.
            reserved=constructors+requirement_witnesses+reserved
        selected_views=reserved+[v for v in selected_views if v not in reserved]
        # A reserved window must not consume the source budget a second time.
        unique_views = {}
        for view in selected_views:
            key = view['view_id']
            if key not in unique_views or view.get('contract_anchor'):
                unique_views[key] = view
        selected_views = list(unique_views.values())
        diag['requirement_candidates']=coverage
        from src.paddleocr_context_selection import select_context
        chosen=select_context(selected_views,
            symbols=[s for req in plan['requirements'] for s in req.get('symbols',[])],
            top_k=top_k,max_windows=max_windows_per_source)
        selected_parents={v['parent_chunk_id'] for v in chosen}
        diag['context_requirement_reservations']={rid:cid in selected_parents if cid else False for rid,cid in coverage.items()}
        hits=materialize_evidence(self.index.chunks,chosen,applicability=self.applicability)
        for rank,hit in enumerate(hits,1):
            hit.update(rank=rank,retrieval_policy=diag['actual_strategy'],
                       retrieval_score=max(v['rank_score'] for v in chosen if v['parent_chunk_id']==hit['chunk_id']),
                       score_type=diag.get('score_type','window_rrf_rank_score'))
        diag.update(final_evidence_count=len(hits),selected_windows=len(chosen),status='OK' if hits else 'NO_CANDIDATES')
        return {'results':hits,'requirements':plan['requirements'],'diagnostics':diag,'candidate_evidence':candidate_evidence}
