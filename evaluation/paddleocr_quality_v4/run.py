"""New grouped questions, frozen before strategy selection; no label input to search."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'versioned-rag-service'))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_query_plan import plan_query
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_retrieval_views import evidence_text, generation_evidence_text
from src.paddleocr_impact_evaluation import assess_evidence, aggregate

FOLDER = Path(__file__).parent/os.getenv('PADDLEOCR_EXPERIMENT_REVISION','.')
# New query wording and task groups; not independent business-expert annotation.
CASES = [
 ('dev','OCR.md','OCR 产线的 rec_texts 和 rec_scores 分别保存什么，是否经过过滤？',['rec_texts','rec_scores','text_rec_score_thresh']),
 ('dev','OCR.md','OCR 结果如何调用 save_to_json 保存到目录？',['save_to_json','save_path']),
 ('dev','OCR.md','OCR 产线 predict 返回结果后，怎么逐个打印并保存图片？',['for res in','print','save_to_img']),
 ('dev','OCR.md','OCR 产线用哪些开关关闭文档方向、矫正与文本行方向？',['use_doc_orientation_classify','use_doc_unwarping','use_textline_orientation']),
 ('dev','OCR.md','OCR 的 text_det_box_thresh 是像素阈值还是框内平均得分阈值？',['text_det_box_thresh','平均']),
 ('dev','OCR.md','OCR 结果中 rec_boxes 的形状和坐标顺序是什么？',['rec_boxes','x_min','y_max']),
 ('dev','PP-StructureV3.md','PPStructureV3 如何导出 Markdown 文档？',['save_to_markdown']),
 ('dev','PP-StructureV3.md','PPStructureV3 的 overall_ocr_res 中识别文本和得分字段是什么？',['overall_ocr_res','rec_texts','rec_scores']),
 ('dev','PP-StructureV3.md','PPStructureV3 是否能通过 use_table_recognition 控制表格识别？',['use_table_recognition']),
 ('dev','PP-StructureV3.md','PPStructureV3 结果如何合并为一个 Markdown 文件？',['concatenate_markdown_pages','save_to_markdown']),
 ('dev','OCR.md','OCR 产线的 cpu_threads 参数类型和默认值是什么？',['cpu_threads','8']),
 ('dev','OCR.md','OCR 的 json 和 img 属性分别返回哪种数据？',['json','img','dict']),
 ('holdout','doc_preprocessor.md','文档预处理未启用方向分类时 angle 的值是什么？',['angle','-1']),
 ('holdout','doc_preprocessor.md','文档预处理能输出哪些角度分类结果？',['angle','0,90,180,270']),
 ('holdout','doc_preprocessor.md','文档预处理保存 JSON 时 numpy.array 如何处理？',['save_to_json','numpy.array','列表']),
 ('holdout','doc_preprocessor.md','文档预处理同时处理多张图，保存图片指定单个文件有什么后果？',['save_to_img','覆盖']),
 ('holdout','table_recognition_v2.md','表格识别产线如何保存 HTML 格式的结果？',['save_to_html']),
 ('holdout','table_recognition_v2.md','表格识别产线的 img 属性有哪些可视化图像？',['table_res_img','ocr_res_img','layout_res_img','preprocessed_img']),
 ('holdout','table_recognition_v2.md','表格识别产线中 rec_texts 和 rec_scores 是什么？',['rec_texts','rec_scores']),
 ('holdout','table_recognition_v2.md','表格识别产线如何指定有线表格单元格检测模型和模型目录？',['wired_table_cells_detection_model_name','wired_table_cells_detection_model_dir']),
]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True)
    if (FOLDER/'freeze.json').exists(): raise RuntimeError('Already frozen; do not overwrite')
    index = PublicKnowledgeIndex()
    rows = []
    for number,(split,filename,query,tokens) in enumerate(CASES,1):
        path = 'docs/version3.x/pipeline_usage/'+filename
        candidates = [h for h in index.chunks if h['version']=='v3.0.0' and h['document_path']==path
                      and all(t in h['content'] for t in tokens)]
        # A compound fact can need multiple canonical source blocks.
        facts = []
        for token_group in ([tokens] if candidates else [[t] for t in tokens]):
            hits = [h for h in index.chunks if h['version']=='v3.0.0' and h['document_path']==path
                    and all(t in h['content'] for t in token_group)]
            if not hits: raise ValueError(f'Missing source label: {number}, {token_group}')
            from src.paddleocr_retrieval_views import module_for
            facts.append({'allowed_paths':[path], 'module':module_for(hits[0]), 'tokens':token_group})
        rows.append({'id':f'v4-{number:02}', 'split':split, 'group':path, 'version':'v3.0.0', 'query':query, 'facts':facts})
    (FOLDER/'dataset.json').write_text(json.dumps({'label_origin':'developer-authored source-grounded diagnostic sample; not independent expert evaluation',
        'split_method':'pipeline document groups; holdout opened only after development selection; no V3 sealed retuning',
        'cases':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    tracked = ['versioned-rag-service/public_corpus_paddleocr/chunks.json',
               'versioned-rag-service/src/paddleocr_query_plan.py',
               'versioned-rag-service/src/paddleocr_retrieval_views.py',
               'versioned-rag-service/src/paddleocr_evidence_search.py',
               'versioned-rag-service/src/paddleocr_fact_checks.py',
               'versioned-rag-service/src/public_api.py',
               'versioned-rag-service/src/claim_support.py',
               'versioned-rag-service/src/paddleocr_impact_evaluation.py',
               'versioned-rag-service/src/public_knowledge.py',
               'versioned-rag-service/src/paddleocr_rag_release.py',
               'versioned-rag-service/src/answer_generation.py',
               'evaluation/paddleocr_quality_v4/run.py',
               str((FOLDER/'dataset.json').relative_to(ROOT)).replace('\\','/')]
    (FOLDER/'freeze.json').write_text(json.dumps({'files':{p:digest(ROOT/p) for p in tracked},
        'top_k':5,'candidate_budget':40,'selection':'maximize dev complete, then minimize wrong modules; no holdout tuning'},indent=2)+'\n',encoding='utf8')

def run(split, name):
    frozen=json.loads((FOLDER/'freeze.json').read_text())
    if any(digest(ROOT/p)!=h for p,h in frozen['files'].items()): raise RuntimeError('Frozen implementation/dataset changed')
    output_path=FOLDER/f'{name}-{split}.json'
    if output_path.exists(): raise RuntimeError('Do not overwrite experiment outputs')
    index=PublicKnowledgeIndex()
    data=json.loads((FOLDER/'dataset.json').read_text(encoding='utf8'))
    cases=[c for c in data['cases'] if c['split']==split]
    ranker=None
    if name=='llm':
        from src.answer_generation import StructuredAnswerGenerator
        ranker=StructuredAnswerGenerator(provider='deepseek',model=os.getenv('RD_V2_GENERATION_MODEL','deepseek-flash')).rerank_candidate_ids
    strategy={'legacy':'bm25','bm25':'contextual_bm25','minilm':'contextual_rerank','bge':'contextual_rerank',
              'rrf':'contextual_rrf_rerank','llm':'contextual_llm_rerank'}[name]
    search=None if name=='legacy' else configured_evidence_search(index,strategy)
    rows=[]
    for c in cases:
        start=time.perf_counter()
        if name=='legacy':
            hits=[h for h in index.search(c['query'],version=c['version'],language='zh',top_k=5) if h.get('retrieval_score',0)>0];diag={}
        else:
            result=search.search(plan_query(c['query'],versions=(c['version'],)),top_k=5,candidate_budget=40,strategy=strategy,ranker=ranker)
            hits,diag=result['results'],result['diagnostics']
        assembled=[{**h,'content':generation_evidence_text(h)} for h in hits]
        for h in assembled: h.pop('evidence_spans',None)
        row={'id':c['id'],'score':assess_evidence(c,hits),'assembled_score':assess_evidence(c,assembled),
             'latency_ms':round((time.perf_counter()-start)*1000,2),
             'diagnostics':diag,'hits':[{k:h[k] for k in ('chunk_id','version','document_path')}|
                {'text':evidence_text(h),'generation_text':generation_evidence_text(h),
                 **({'evidence_spans':h['evidence_spans']} if 'evidence_spans' in h else {})} for h in hits]}
        rows.append(row)
        print(c['id'],row['score']['complete'],round(row['latency_ms']),flush=True)
    output={'strategy':name,'split':split,'freeze_sha256':digest(FOLDER/'freeze.json'),
            'metrics':aggregate([r['score'] for r in rows]),'assembled_metrics':aggregate([r['assembled_score'] for r in rows]),'rows':rows}
    output_path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output['metrics']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--split',choices=['dev','holdout'],default='dev')
    p.add_argument('--strategy',choices=['legacy','bm25','minilm','bge','rrf','llm'],default='bm25');args=p.parse_args()
    if args.prepare: prepare()
    else: run(args.split,args.strategy)
