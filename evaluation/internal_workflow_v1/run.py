"""Bounded source-grounded selection; holdout never participates in selection."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versioned-rag-service'))
from src.public_knowledge import PublicKnowledgeIndex
from src.paddleocr_query_plan import plan_query
from src.paddleocr_evidence_search import configured_evidence_search
from src.paddleocr_retrieval_views import module_for,evidence_text,generation_evidence_text
from src.paddleocr_impact_evaluation import assess_evidence,aggregate
from src.paddleocr_internal_release import neural_execution_succeeded
FOLDER=Path(__file__).parent/'final'
CASES=[
 ('dev','v2.9.1','doc/doc_ch/whl.md','中文 OCR 初始化如何启用角度分类？',['PaddleOCR','use_angle_cls=True','lang="ch"']),
 ('dev','v2.9.1','doc/doc_ch/whl.md','OCR 已有裁剪图时如何只识别，不进行检测？',['ocr.ocr','det=False']),
 ('dev','v2.9.1','doc/doc_ch/whl.md','OCR 只需要检测文字框时怎样关闭识别？',['ocr.ocr','rec=False']),
 ('dev','v2.9.1','doc/doc_ch/whl.md','命令行输入 PDF 时如何限制处理页数？',['page_num','默认为0']),
 ('dev','v2.9.1','paddleocr.py','OCR 旧版本如何组合文字框和识别结果返回一页？',['ocr_res.append','zip(dt_boxes, rec_res)']),
 ('dev','v3.0.0','paddleocr/_pipelines/ocr.py','OCR 的 ocr 入口内部是否调用 predict？',['def ocr','self.predict']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 结果的文本与识别得分字段名称是什么？',['rec_texts','rec_scores']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 怎样导出 JSON 结果？',['save_to_json','save_path']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 怎样跳过文档矫正和方向分类？',['use_doc_orientation_classify','use_doc_unwarping','use_textline_orientation']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR text_det_box_thresh 如何决定保留检测框？',['text_det_box_thresh','平均']),
 ('dev','v3.0.0','docs/update/upgrade_notes.md','原来的 PPStructure 接口在新版中如何迁移？',['PPStructure','PPStructureV3']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 可视化识别框 rec_boxes 采用什么坐标？',['rec_boxes','x_min','y_max']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版 drop_score 的作用及默认值是什么？',['drop_score','0.5','丢弃']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版 cpu_threads 的默认线程数量是多少？',['cpu_threads','10']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版 det_limit_type 可选类型如何解释？',['det_limit_type','min','max']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版 use_onnx 参数控制什么？',['use_onnx','onnx']),
 ('validation','v3.0.0','docs/version3.x/installation.md','安装时要求的飞桨框架最低版本是什么？',['3.0','飞桨框架']),
 ('validation','v3.0.0','docs/version3.x/installation.md','只做推理需安装哪个 pip 包？',['pip install paddleocr']),
 ('validation','v3.0.0','docs/version3.x/installation.md','训练依赖如何从 requirements 文件安装？',['pip install -r requirements.txt']),
 ('validation','v3.0.0','paddleocr/_common_args.py','precision 遇到不支持的值时如何处理？',['precision','ValueError','SUPPORTED_PRECISION_LIST']),
 ('holdout','v2.9.1','tests/test_paddleocr_api.py','官方测试如何验证仅检测的 OCR 调用？',['test_ocr_det_only','det=True, rec=False','isinstance(result, list)']),
 ('holdout','v2.9.1','tests/test_paddleocr_api.py','官方测试如何验证仅识别的 OCR 调用？',['test_ocr_rec_only','det=False, rec=True','isinstance(result, list)']),
 ('holdout','v3.0.0','tests/pipelines/test_ocr.py','官方 OCR predict 测试如何检查每页识别文本类型？',['test_predict','rec_texts','isinstance(text, str)']),
 ('holdout','v3.0.0','tests/pipelines/test_ocr.py','官方测试怎样给中文 PP-OCRv4 匹配识别模型名？',['lang="ch", ocr_version="PP-OCRv4"','PP-OCRv4_mobile_rec']),
 ('holdout','v3.0.0','docs/version3.x/pipeline_usage/doc_preprocessor.md','文档预处理关闭角度分类时 angle 如何表示？',['angle','-1']),
 ('holdout','v3.0.0','docs/version3.x/pipeline_usage/doc_preprocessor.md','文档预处理 json 保存时 numpy.array 如何转换？',['save_to_json','numpy.array','列表']),
 ('holdout','v3.0.0','docs/version3.x/pipeline_usage/doc_preprocessor.md','文档预处理批量保存到一个图片文件路径有什么风险？',['save_to_img','覆盖']),
 ('holdout','v3.0.0','docs/version3.x/pipeline_usage/doc_preprocessor.md','文档预处理方向输出可能有哪些角度？',['angle','0,90,180,270']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','文字检测框膨胀系数使用哪个参数？',['text_det_unclip_ratio']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 如何设置识别文本最低得分？',['text_rec_score_thresh']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','文字识别批大小如何设置？',['text_recognition_batch_size']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','文本行方向分类的批大小如何设置？',['text_line_orientation_batch_size']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 输入是否接受 numpy 图像数组？',['numpy.ndarray']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','旧的 det_db_unclip_ratio 迁移为什么参数？',['det_db_unclip_ratio','text_det_unclip_ratio']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','文字识别输出的四点多边形字段是什么？',['rec_polys']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','文字检测输出的多边形字段是什么？',['dt_polys']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','OCR 结果如何保存可视化图片？',['save_to_img','save_path']),
 ('dev','v3.0.0','docs/version3.x/pipeline_usage/OCR.md','怎样在初始化时指定运行设备？',['device']),
 ('dev','v2.9.1','doc/doc_ch/whl.md','旧版 OCR 的语言参数如何表示中文？',['lang','中英文(ch)']),
 ('dev','v2.9.1','paddleocr.py','旧 OCR 接口输入列表但开启检测时会怎样？',['isinstance(img, list)','det','exit(0)']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版是否保存识别裁剪图由哪个参数控制？',['save_crop_res','False']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','识别裁剪图默认保存在哪里？',['crop_res_save_dir','./output']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版使用 TensorRT 推理默认启用吗？',['use_tensorrt','False']),
 ('validation','v2.9.1','doc/doc_ch/inference_args.md','旧版推理 precision 的默认值是什么？',['precision','fp32']),
 ('holdout','v3.0.0','docs/version3.x/module_usage/text_recognition.md','文本识别模块结果中得分字段是什么？',['rec_score','置信度']),
 ('holdout','v3.0.0','docs/version3.x/module_usage/text_recognition.md','文本识别模块是否支持 numpy 数组输入？',['numpy.ndarray']),
 ('holdout','v3.0.0','docs/version3.x/module_usage/text_recognition.md','文本识别 predict_iter 与 predict 的返回方式有什么区别？',['predict_iter','generator']),
 ('holdout','v3.0.0','docs/version3.x/module_usage/text_recognition.md','文本识别模块如何将结果存为 JSON？',['save_to_json','save_path']),
]
CONFIGS={
 'legacy':{'strategy':'legacy','top_k':5,'window_budget':384,'candidate_budget':40,'use_context':False},
 'scoped_256':{'strategy':'contextual_bm25','top_k':5,'window_budget':256,'candidate_budget':40,'use_context':True},
 'scoped_384':{'strategy':'contextual_bm25','top_k':5,'window_budget':384,'candidate_budget':40,'use_context':True},
 'scoped_384_k8':{'strategy':'contextual_bm25','top_k':8,'window_budget':384,'candidate_budget':80,'use_context':True},
 'plain_384':{'strategy':'contextual_bm25','top_k':5,'window_budget':384,'candidate_budget':40,'use_context':False},
 'bge_rerank':{'strategy':'contextual_rerank','top_k':5,'window_budget':384,'candidate_budget':40,'use_context':True},
}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    p=FOLDER/name
    if p.exists():raise ValueError('preserve frozen artifact: '+name)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def read(name):return json.loads((FOLDER/name).read_text(encoding='utf8'))


def prepare():
    FOLDER.mkdir(exist_ok=True)
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    rows=[]
    for i,(split,version,path,query,tokens) in enumerate(CASES,1):
        hits=[h for h in index.chunks if h['version']==version and h['document_path']==path]
        groups=[tokens] if any(all(t in h['content'] for t in tokens) for h in hits) else [[t] for t in tokens]
        facts=[]
        for group in groups:
            matching=[h for h in hits if all(t in h['content'] for t in group)]
            if not matching:raise ValueError(f'label not in source: {i} {group}')
            facts.append({'allowed_paths':[path],'module':module_for(matching[0]),'tokens':group})
        rows.append({'id':f'internal-{i:02}','split':split,'group':version+':'+path,'version':version,'query':query,'facts':facts})
    write('dataset.json',{'cases':rows,'label_origin':'developer source-grounded diagnostic; not independent expert or production accuracy',
        'split_method':'disjoint version/document groups; preceding 28 questions already opened during diagnostic fixes; 20 additions frozen before this run. This is a grouped regression set, not an independent blind test.'})
    paths=[ROOT/'versioned-rag-service/src'/n for n in ('paddleocr_evidence_search.py','paddleocr_retrieval_views.py','paddleocr_query_plan.py','paddleocr_quality.py','paddleocr_rag_release.py','paddleocr_internal_release.py','public_api.py','public_knowledge.py','paddleocr_impact_evaluation.py','paddleocr_compatibility.py','paddleocr_application_roles.py')]
    paths += [p for p in (ROOT/'versioned-rag-service/public_corpus_paddleocr').rglob('*') if p.is_file()]
    paths += [Path(__file__),FOLDER/'dataset.json',ROOT/'change-review-agent/app/paddleocr_investigation.py']
    write('freeze.json',{'files':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths},
        'configs':CONFIGS,'selection_rule':'rank by dev complete coverage, fewer wrong modules, fewer evidence blocks; choose first candidate passing validation against legacy, otherwise retain legacy. Open holdout once without reselection. No latency objective.',
        'neural_candidate':'local BGE reranker only; no new download; fallback is not a successful neural experiment'})


def verify():
    for name,sha in read('freeze.json')['files'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen source changed: '+name)


def run(split,name):
    verify();cfg=CONFIGS[name]
    if split=='holdout' and not (FOLDER/'selection.json').exists():raise ValueError('select on development/validation first')
    index=PublicKnowledgeIndex(ROOT/'versioned-rag-service/public_corpus_paddleocr')
    search=None if cfg['strategy']=='legacy' else configured_evidence_search(index,cfg['strategy'],use_context=cfg['use_context'],window_budget=cfg['window_budget'])
    rows=[]
    for case in [c for c in read('dataset.json')['cases'] if c['split']==split]:
        started=time.perf_counter()
        if search is None:
            hits=[h for h in index.search(case['query'],version=case['version'],language='zh',top_k=cfg['top_k']) if h.get('retrieval_score',0)>0];diag={'actual_strategy':'legacy'}
        else:
            output=search.search(plan_query(case['query'],versions=(case['version'],)),top_k=cfg['top_k'],candidate_budget=cfg['candidate_budget'],strategy=cfg['strategy'])
            hits=output['results'];diag=output['diagnostics']
        rows.append({'id':case['id'],'score':assess_evidence(case,hits),'latency_ms':round((time.perf_counter()-started)*1000,2),'diagnostics':diag,
            'hits':[{k:h[k] for k in ('chunk_id','version','document_path')}|{'text':evidence_text(h),'generation_text':generation_evidence_text(h),**({'evidence_spans':h['evidence_spans']} if 'evidence_spans' in h else {})} for h in hits]})
        print(name,case['id'],rows[-1]['score']['complete'],flush=True)
    write(f'{name}-{split}.json',{'strategy':name,'split':split,'freeze_sha256':digest(FOLDER/'freeze.json'),
        'metrics':aggregate([r['score'] for r in rows]),'rows':rows})


def select():
    verify();eligible=[]
    for name in CONFIGS:
        report=read(f'{name}-dev.json')
        if name=='bge_rerank' and not neural_execution_succeeded(report['rows']):continue
        m=report['metrics']
        if name!='legacy' and m['wrong_version_count']==0:
            eligible.append((name,m))
    ranking=sorted(eligible,key=lambda x:(-x[1]['complete_count'],x[1]['wrong_module_count'],CONFIGS[x[0]]['top_k'],x[0]))
    name=ranking[0][0]
    baseline=read('legacy-validation.json')['metrics']
    def passes(n):
        report=read(n+'-validation.json')
        if n=='bge_rerank' and not neural_execution_succeeded(report['rows']):return False
        candidate=report['metrics']
        return candidate['complete_count']>=baseline['complete_count'] and candidate['wrong_version_count']==0 and candidate['wrong_module_count']<=baseline['wrong_module_count']
    qualified=[n for n,m in ranking if passes(n)]
    passed=bool(qualified)
    write('selection.json',{'development_candidate':name,'selected':qualified[0] if passed else 'legacy',
        'validation_gate_passed':passed,'holdout_status_at_selection':'NOT_OPENED','rule':read('freeze.json')['selection_rule']})
    print(read('selection.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','select']);parser.add_argument('--split',default='dev',choices=['dev','validation','holdout']);parser.add_argument('--name',default='scoped_384',choices=list(CONFIGS));args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='select':select()
    else:run(args.split,args.name)
