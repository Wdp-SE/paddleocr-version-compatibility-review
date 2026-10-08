"""Optional neural retrieval. Original evidence is never rewritten by ranking."""
from __future__ import annotations
import hashlib
import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .retrieval_fusion import fuse_ranked_hits

STRATEGIES = ('bm25', 'semantic', 'bm25_rerank', 'hybrid', 'hybrid_rerank')


class ModelUnavailable(RuntimeError):
    pass

class CachedEncoder:
    """Reuse exact document representations, never query labels or answers."""
    def __init__(self, encode, *, identity, cache_root=''):
        self.encode=encode;self.identity=identity;self.root=Path(cache_root) if cache_root else None
        self.lock=threading.Lock()

    def __call__(self,texts):
        with self.lock:
            path=None
            if self.root and len(texts)>1:
                key=hashlib.sha256(json.dumps([self.identity,texts],ensure_ascii=False).encode()).hexdigest()
                path=self.root/('views-'+key+'.npy')
                if path.is_file():
                    values=np.load(path,allow_pickle=False)
                    if values.ndim!=2 or len(values)!=len(texts) or not np.isfinite(values).all():
                        raise ModelUnavailable('SEMANTIC_CACHE_INVALID')
                    return values
            values=np.asarray(self.encode(texts),dtype=np.float32)
            if values.ndim!=2 or len(values)!=len(texts) or not np.isfinite(values).all():
                raise ModelUnavailable('SEMANTIC_NONFINITE_OUTPUT')
            if path:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
                try:
                    with temporary.open('wb') as output:np.save(output,values,allow_pickle=False)
                    temporary.replace(path)
                finally:temporary.unlink(missing_ok=True)
            return values


class ExactScoreCache:
    """Model-instance-local, bounded cache of raw query/text logits; no labels."""
    def __init__(self,predict,*,max_pairs=20000):
        self.predict=predict;self.max_pairs=max_pairs;self.cache=OrderedDict();self.lock=threading.Lock()

    def __call__(self,pairs):
        keys=[hashlib.sha256((q+'\0'+t).encode('utf-8')).hexdigest() for q,t in pairs]
        with self.lock:
            missing={k:pair for k,pair in zip(keys,pairs) if k not in self.cache}
            fresh={}
            if missing:
                scores=np.asarray(self.predict(list(missing.values()))).reshape(-1)
                if len(scores)!=len(missing) or not np.isfinite(scores).all():
                    raise ModelUnavailable('RERANK_INVALID_OUTPUT')
                fresh=dict(zip(missing,map(float,scores)))
            values=[fresh[k] if k in fresh else self.cache[k] for k in keys]
            self.cache.update(fresh)
            for k in keys:
                if k in self.cache:self.cache.move_to_end(k)
            while len(self.cache)>self.max_pairs:self.cache.popitem(last=False)
            return values


class QualityRetriever:
    def __init__(self, index, *, encoder=None, vectors=None, scorer=None):
        self.index = getattr(index, 'base_index', index)
        if self.index.manifest.get('workspace_id') != 'paddleocr':
            raise ValueError('quality retrieval is only supported for PaddleOCR')
        self.encoder, self.vectors, self.scorer = encoder, vectors, scorer

    def search(self, query, *, version='current', language='zh_preferred', strategy='hybrid_rerank', top_k=5, candidate_k=20):
        if strategy not in STRATEGIES or not 1 <= top_k <= 20 or not 1 <= candidate_k <= 20:
            raise ValueError('invalid quality retrieval request')
        # Delegate validation to the same source-bound index used by lexical search.
        lexical = [hit for hit in self.index.search(query, version=version, language=language, policy='bm25', top_k=candidate_k)
                   if hit.get('retrieval_score',0)>0]
        if strategy == 'bm25':
            return lexical[:top_k]
        semantic = []
        if strategy in ('semantic', 'hybrid', 'hybrid_rerank'):
            if self.encoder is None or self.vectors is None:
                raise ModelUnavailable('SEMANTIC_MODEL_NOT_CONFIGURED')
            members = self.index._version_members(version)
            eligible = [i for i,c in enumerate(self.index.chunks)
                        if (members is None or c['version'] in members)
                        and (language not in ('zh','en') or c['language']==language)]
            q = np.asarray(self.encoder(['为这个句子生成表示以用于检索相关文章：'+query]),dtype=np.float32)
            vectors = np.asarray(self.vectors,dtype=np.float32)
            if (q.ndim != 2 or q.shape[0]!=1 or vectors.shape!=(len(self.index.chunks),q.shape[1])
                    or not np.isfinite(q).all() or not np.isfinite(vectors).all()):
                raise ModelUnavailable('SEMANTIC_INVALID_OUTPUT')
            scores=vectors[eligible] @ q[0]
            order=sorted(range(len(eligible)),key=lambda j:(-float(scores[j]),eligible[j]))[:candidate_k]
            semantic=[{**self.index.chunks[eligible[j]],'retrieval_score':float(scores[j]),
                       'rank':rank,'retrieval_policy':'bge_small_zh','score_type':'embedding_cosine'}
                      for rank,j in enumerate(order,1)]
        if strategy=='semantic':
            return semantic[:top_k]
        candidates=(fuse_ranked_hits([lexical,semantic],top_k=candidate_k*2,rrf_k=60)
                    if semantic else lexical)
        if strategy=='hybrid':
            return candidates[:top_k]
        if not candidates:
            return []
        if self.scorer is None:
            raise ModelUnavailable('RERANK_MODEL_NOT_CONFIGURED')
        scores=np.asarray(self.scorer([(query,c['content']) for c in candidates])).reshape(-1)
        if len(scores)!=len(candidates) or not np.isfinite(scores).all():
            raise ModelUnavailable('RERANK_INVALID_OUTPUT')
        order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),i))[:top_k]
        return [{**candidates[i], 'rank':rank, 'rerank_score':float(scores[i]),'candidate_pool_size':len(candidates),
                 'score_type':'cross_encoder_logit','retrieval_policy':strategy}
                for rank,i in enumerate(order,1)]


_LOCK=threading.Lock()
_CACHE={}
_ASSET_CACHE={}
_MODELS={}


def configure_cpu_inference(torch):
    # This Windows CPU stack produced drifting embeddings and nonfinite logits
    # with optimized parallel kernels. Native serial inference was repeatable.
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False


def asset_identity(folder):
    path=Path(folder)
    files=sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.safetensors','.bin') and '.cache' not in p.relative_to(path).parts) if folder else []
    signature=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files)
    if signature not in _ASSET_CACHE:
        _ASSET_CACHE[signature]=hashlib.sha256(''.join(str(p.relative_to(path))+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest() if files else 'NOT_CONFIGURED'
    return _ASSET_CACHE[signature]


def configured_quality_retriever(index, *, load_embedding=True, load_reranker=True, build_parent_vectors=True):
    """Load local assets only; HTTP requests never trigger model downloads."""
    base=getattr(index,'base_index',index)
    embedding=os.environ.get('PADDLEOCR_EMBEDDING_PATH','').strip()
    reranking=os.environ.get('PADDLEOCR_RERANKER_PATH','').strip()
    if not load_embedding:embedding=''
    if not load_reranker:reranking=''
    if not embedding and not reranking:
        raise ModelUnavailable('QUALITY_MODELS_NOT_CONFIGURED')
    fingerprint=hashlib.sha256(''.join(c['chunk_id']+c['content'] for c in base.chunks).encode()).hexdigest()
    key=(fingerprint,embedding,reranking,build_parent_vectors,asset_identity(embedding),asset_identity(reranking))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        try:
            import torch
            configure_cpu_inference(torch)
            from sentence_transformers import SentenceTransformer, CrossEncoder
            encoder=vectors=scorer=None
            if embedding:
                if not Path(embedding).is_dir(): raise ModelUnavailable('SEMANTIC_ASSET_NOT_FOUND')
                # Eager attention avoids NaN embeddings observed with this CPU Torch/SDPA stack.
                embedding_key=('embedding',embedding,key[-2],'native-serial-v2')
                if embedding_key not in _MODELS:
                    model=SentenceTransformer(embedding,device='cpu',local_files_only=True,
                                              model_kwargs={'attn_implementation':'eager'})
                    encoder=CachedEncoder(lambda texts: model.encode(texts,normalize_embeddings=True,batch_size=8,show_progress_bar=False),
                                          identity=key[-2]+'native-serial-v2',cache_root=os.environ.get('PADDLEOCR_VECTOR_CACHE',''))
                    _MODELS[embedding_key]=(encoder,model.tokenizer,model.get_sentence_embedding_dimension())
                encoder,embedding_tokenizer,embedding_dimension=_MODELS[embedding_key]
                cache_root=os.environ.get('PADDLEOCR_VECTOR_CACHE','').strip()
                cache_path=None
                if cache_root:
                    asset_hash=key[-2]
                    cache_key=hashlib.sha256((fingerprint+asset_hash+'eager-native-serial-v2').encode()).hexdigest()
                    cache_path=Path(cache_root)/(cache_key+'.npy')
                if not build_parent_vectors:
                    vectors=None
                elif cache_path and cache_path.is_file():
                    vectors=np.load(cache_path,allow_pickle=False)
                    if vectors.shape!=(len(base.chunks),embedding_dimension):
                        raise ModelUnavailable('SEMANTIC_CACHE_SHAPE_MISMATCH')
                else:
                    vectors=encoder([c['content'] for c in base.chunks])
                if vectors is not None and not np.isfinite(vectors).all():raise ModelUnavailable('SEMANTIC_NONFINITE_OUTPUT')
                if vectors is not None and cache_path and not cache_path.is_file():
                    cache_path.parent.mkdir(parents=True,exist_ok=True)
                    np.save(cache_path,vectors,allow_pickle=False)
            if reranking:
                if not Path(reranking).is_dir(): raise ModelUnavailable('RERANK_ASSET_NOT_FOUND')
                rerank_key=('rerank',reranking,key[-1],'native-serial-v2')
                if rerank_key not in _MODELS:
                    model_rank=CrossEncoder(reranking,device='cpu',local_files_only=True,max_length=512,
                                            model_kwargs={'attn_implementation':'eager'})
                    # Return raw logits, not sigmoid probabilities.
                    scorer=ExactScoreCache(lambda pairs: model_rank.predict(pairs,batch_size=2,activation_fn=torch.nn.Identity(),show_progress_bar=False))
                    _MODELS[rerank_key]=(scorer,model_rank.tokenizer)
                scorer,rerank_tokenizer=_MODELS[rerank_key]
            retriever=QualityRetriever(base,encoder=encoder,vectors=vectors,scorer=scorer)
            retriever.model_identity={'embedding':key[-2],'reranker':key[-1],'attention':'eager','cpu_kernel':'native_serial_v2'}
            tokenizers=[]
            if embedding:tokenizers.append(embedding_tokenizer)
            if reranking:tokenizers.append(rerank_tokenizer)
            retriever.token_count=lambda text: max(len(t.encode(text,add_special_tokens=True,truncation=False)) for t in tokenizers)
        except ModelUnavailable:
            raise
        except (ImportError,OSError,ValueError,RuntimeError) as exc:
            raise ModelUnavailable('QUALITY_MODEL_LOAD_FAILED') from exc
        _CACHE[key]=retriever
        return retriever
