"""Conservative query requirements; rewrite proposals cannot change source scope."""
from __future__ import annotations
import re
from src.paddleocr_corpus import PADDLEOCR_VERSIONS

_VERSION=re.compile(r'(?<![\w.])v?\d+\.\d+\.\d+(?![\w.])',re.I)
_SYMBOL=re.compile(r'\b(?:[A-Za-z][\w]*_[\w]+|PaddleOCR|PPStructureV3|PPStructure)\b')
_MODULES={'文本检测':'text_detection','文本识别':'text_recognition',
          '方向分类':'doc_img_orientation_classification','图像矫正':'text_image_unwarping',
          '表格结构':'table_structure_recognition','版面检测':'layout_detection',
          'PPStructureV3':'structure','PPStructure':'structure','文档预处理':'doc_preprocessor'}
_COMPONENT_APIS={'TextDetection':'text_detection','TextRecognition':'text_recognition',
                 'TextLineOrientationClassification':'text_line_orientation_classification',
                 'DocImgOrientationClassification':'doc_img_orientation_classification',
                 'TextImageUnwarping':'text_image_unwarping','LayoutDetection':'layout_detection',
                 'TableStructureRecognition':'table_structure_recognition',
                 'TableClassification':'table_classification','TableCellsDetection':'table_cells_detection'}
_MODULES.update(_COMPONENT_APIS)
REQUIREMENT_BUDGET=8
_DIMENSIONS=(
    ('call','调用入口',r'调用入口|调用方式|(?:怎么|如何|怎样)调用|(?:执行|运行|推理)(?:什么|哪个)?(?:函数|接口|API)'),
    ('parameters','初始化参数',r'初始化(?:参数)?|构造参数|构造器|构造函数|实例化|(?:创建|构建)(?:对象|实例)|(?:对象|实例).{0,4}(?:创建|构建)'),
    ('result','结果读取',r'结果读取|结果归一化|返回结构|(?:结果|输出|返回值)(?:怎么|如何|怎样)读|从返回值.{0,5}(?:取|拿)'),
)


def target_missing_requirements(plan: dict, missing_ids) -> dict:
    """Search only original unmet obligations; model gap text is never a query."""
    import copy
    result = copy.deepcopy(plan)
    wanted = set(missing_ids)
    rows = [r for r in plan['requirements'] if r['id'] in wanted][:4]
    result['repair_requirement_ids'] = [r['id'] for r in rows]
    result['proposed_queries'] = list(dict.fromkeys(r.get('retrieval_query', r['query']) for r in rows))
    return result


def _atomic_clauses(query):
    clauses=[s.strip() for s in re.split(r'[；;\n？?]+|以及',query) if s.strip()]
    result=[]
    for clause in clauses:
        # Formatting/source requests and an explicit scope exclusion are not
        # additional facts. Retain them in original_query but spend no facet slots.
        if re.fullmatch(r'(?:请|并|同时)?(?:分别)?(?:给出|提供|列出|附上)(?:两版|两个版本|各版|对应版本|相关|本版|上述|以上|具体|相应|的|\s)*(?:出处|来源|引用|依据|证据)[。.]?',clause):
            continue
        if re.match(r'^(?:这里|本题|这次)?(?:不是|并非|而非|别拿|别用)',clause) and not re.search(r'什么|如何|怎么|为何|为什么|吗|是否|能否',clause):
            remainder=re.sub(r'(?:不是|并非|而非|别拿|别用)[^，,。；;\n？?]*','',clause)
            remainder=re.sub(r'^(?:这里|本题|这次)','',remainder).strip(' ，,。.')
            if not remainder:
                continue
        dimensions=[(key,label) for key,label,pattern in _DIMENSIONS if re.search(pattern,clause)]
        version_prefix=' / '.join(sorted(explicit_versions(clause)))
        if len(dimensions)>1:
            positions=sorted((match.start(),key) for key,_,pattern in _DIMENSIONS
                for match in re.finditer(pattern,clause))
            for i,(start,key) in enumerate(positions):
                end=positions[i+1][0] if i+1<len(positions) else len(clause)
                detail=clause[start:end].strip(' 、，,和与')
                result.append((f'PaddleOCR {version_prefix} {detail}',key))
        elif retrieval_intent(clause)['kind']=='parameter_migration':
            names=list(dict.fromkeys(s for s in _SYMBOL.findall(clause) if s not in ('PaddleOCR','PPStructure','PPStructureV3')))
            if (len(names)>1 and re.search(r'和|与|、|分别',clause)
                    and not re.search(r'=|同时|互斥|覆盖|优先|顺序|一起|都传',clause)):
                result.extend((f'PaddleOCR {version_prefix} 升级时 {name} 参数应如何处理','parameters') for name in names)
            else:result.append((clause,None))
        else:result.append((clause,None))
    return result


def retrieval_intent(text: str) -> dict:
    """Task aliases for source selection, never an answer or a version override."""
    result_read = (re.search(r'结果|返回', text) and re.search(r'读取|消费|结构|格式|归一化', text)) or (
        re.search(r'文字|文本|结果|返回|\bline\s*\[', text)
        and re.search(r'取(?:文字|文本|结果)|索引|下标|\bline\s*\[', text))
    if result_read and (re.search(r'升级|升到|升至|迁移|差异|变化', text) or len(explicit_versions(text))>1):
        return {'kind': 'result_migration', 'aliases': ['ocr_res append zip dt_boxes rec_res', 'line box txt score', 'rec_texts rec_scores', 'def ocr self predict', 'return list self.predict_iter'],
                'preferred_role': 'result_contract'}
    # Moving a keyword between explicitly named versions is also a migration
    # question, even when the user never writes "升级". Ordinary comparisons
    # of defaults must keep their parameter-definition retrieval route.
    cross_version_transfer = (len(explicit_versions(text)) > 1
        and re.search(r'构造器|构造函数|初始化|constructor|__init__', text, re.I)
        and re.search(r'传入|转发|校验|接受|拒绝|报错|能否|可否', text))
    if (re.search(r'参数|选项', text) and (re.search(r'升级|迁移|废弃|兼容|旧参数|重命名', text) or cross_version_transfer)
            and _SYMBOL.search(text)):
        return {'kind': 'parameter_migration', 'aliases': [
            '_DEPRECATED_PARAM_NAME_MAPPING params kwargs base_params',
            'super __init__ parse_common_args common_args',
            'default_vals unknown_names Unknown argument'], 'preferred_role': 'parameter_definition'}
    if (re.search(r'\bcls\s*=\s*(?:True|False)',text) and 'use_angle_cls' in text
            and re.search(r'执行|启动|分类|生效',text)):
        return {'kind':'parameter_behavior','symbols':['use_angle_cls','cls'],
                'aliases':['use_angle_cls cls self.__call__ text_classifier',
                           'if self.use_angle_cls and cls'], 'preferred_role':'implementation_branch'}
    # Bridge colloquial workload units to parameter-table vocabulary. These are
    # search terms only, learned from the document format, never parameter values.
    if _SYMBOL.search(text) and re.search(r'过滤代码|(?:刚好|恰好|分数.{0,8})等于|边界|保留还是|源码.{0,12}过滤',text):
        symbols=[s for s in _SYMBOL.findall(text) if s!='PaddleOCR']
        return {'kind':'parameter_behavior','symbols':symbols,
                'aliases':[' '.join(symbols)+' if score filter append return'],
                'preferred_role':'implementation_branch'}
    if (re.search(r'批量|一批|batch|(?:一次|每次).{0,12}(?:张|幅)', text, re.I)
            and not (re.search(r'结果|predict|预测',text) and re.search(r'打印|保存|导出',text))):
        stage = '识别' if '识别' in text else '检测' if '检测' in text else ''
        return {'kind': 'parameter_lookup', 'aliases': [f'{stage} batch size 批量',
                'img_list img_num batch_num beg_img_no end_img_no',
                'img_crop_list text_recognizer get_rotate_crop_image'],
                'definition_aliases': [f'{stage} batch size 批量'],
                'preferred_role': 'parameter_definition'}
    if re.search(r'测试|断言|\btest\b|\bassert', text, re.I):
        return {'kind': 'test_contract', 'aliases': ['assert result isinstance'],
                'preferred_role': 'test_assertion'}
    if re.search(r'打印|\bprint\b|保存|导出',text) and re.search(r'结果|predict|预测',text):
        methods=[]
        if re.search(r'打印|\bprint\b',text):methods.append('print')
        if re.search(r'(?:保存|导出).{0,6}(?:图片|图像)|save_to_img',text):methods.append('save_to_img')
        if re.search(r'json|JSON|save_to_json',text):methods.append('save_to_json')
        return {'kind':'result_usage','aliases':['predict for res result '+' '.join(methods)],
                'methods':methods,'preferred_role':'api_example'}
    if (re.search(r'文本|文字|字|rec_texts',text) and re.search(r'分数|得分|置信度|rec_scores',text)
            and re.search(r'字段|放哪|保存|哪里|是什么|含义|意思|各|分别',text)):
        container='overall_ocr_res ' if 'overall_ocr_res' in text else ''
        return {'kind':'result_fields','aliases':[container+'rec_texts rec_scores text_rec_score_thresh'],
                'preferred_role':'result_contract'}
    method_lookup = bool(re.search(r'PaddleOCR|(?<![\w-])OCR(?!\w)',text) and query_module(text) == 'ocr'
                         and re.search(r'\bpredict\b|\bocr\s*(?:\(|方法|接口|入口)', text))
    if (re.search(r'调用|接口|Python|参数|裁剪|裁出|裁好|找框|怎么调|该怎么调|\bocr\s*\(', text, re.I) or method_lookup) and not re.search(r'服务|部署|命令行', text):
        aliases = []
        if method_lookup:
            aliases += ['PaddleOCR predict input', 'def ocr self predict']
        if re.search(r'只.{0,8}识别|关闭.{0,8}检测|不.{0,8}(?:检测|找框)|已有.{0,6}裁剪|裁剪.{0,10}(?:输入|识别)', text):
            aliases.append('ocr det rec False')
        return {'kind': 'api_usage', 'aliases': aliases, 'method_lookup': method_lookup, 'preferred_role': 'api_example'}
    return {'kind': 'knowledge', 'aliases': [], 'preferred_role': None}

def explicit_versions(text: str) -> set[str]:
    return {'v'+m.group(0).lower().lstrip('v') for m in _VERSION.finditer(text)}


def query_module(text: str) -> str | None:
    # Excluded pipelines are retrieval constraints, not the question's owner.
    # Keep the raw query for retrieval/answering; this normalization is scope-only.
    text = re.sub(r'(?:不是|并非|而非|不要|别拿|别用|排除)[^，,。；;\n？?]*', ' ', text)
    # A named pipeline owns its component options; mentioning the component in
    # that question does not silently switch the API scope to a standalone module.
    for pattern, module in (
        (r'PPStructure(?:V3)?', 'structure'),
        (r'(?<![\w-])OCR(?!\w)|\bPaddleOCR\s*\(', 'ocr'),
        (r'文档预处理|DocPreprocessor', 'doc_preprocessor'),
        (r'表格识别产线|通用表格识别v2|TableRecognitionPipelineV2', 'table_recognition_v2'),
        (r'文本行方向分类|TextLineOrientationClassification', 'text_line_orientation_classification'),
    ):
        if re.search(pattern, text): return module
    # A legacy OCR call switch is not the standalone document-orientation API.
    if ('PaddleOCR' in text and re.search(r'\buse_angle_cls\b|\bocr\b', text)
            and not re.search(r'独立|模块', text)):
        return 'ocr'
    values={v for k,v in _MODULES.items() if k in text}
    # predict is common to standalone modules too. Only use the product's OCR
    # fallback when no explicit component owns the question.
    if ('PaddleOCR' in text and re.search(r'\bpredict\b', text) and not values
            and not re.search(r'独立|模块', text)):
        values.add('ocr')
    # The product name is not a pipeline. Only an explicit API call or standalone
    # OCR denotes the OCR pipeline; PP-OCR model families also name the product.
    if re.search(r'(?<![\w-])OCR(?!\w)|\bPaddleOCR\s*\(', text):
        values.add('ocr')
    if len(values)>1:values.discard('ocr')
    return next(iter(values)) if len(values)==1 else None


def claim_scope_module(text: str, requirements: list[dict]) -> str | None:
    """Descriptions of result fields are not switches to standalone submodules."""
    modules={r['module'] for r in requirements if r.get('module')}
    # The legacy OCR classifier is a capability of its owning call. A sentence
    # beginning with its Chinese description does not name a standalone API.
    named_api=any(name in text for name in _COMPONENT_APIS)
    if (modules=={'ocr'} and re.search(r'\buse_angle_cls\b',text)
            and not named_api and not re.search(r'独立|方向分类模块|PPStructure',text)):
        return 'ocr'
    explicit = re.search(r'PPStructure(?:V3)?|\bPaddleOCR\s*\(|(?<![\w-])OCR(?!\w)|文档预处理|'
                         r'(?:文本检测|文本识别|方向分类|图像矫正|表格结构|版面检测)模块|'
                         r'^\s*(?:文本检测|文本识别|方向分类|图像矫正|表格结构|版面检测)', text)
    if explicit: return query_module(text)
    return next(iter(modules)) if len(modules)==1 else None


def plan_query(query: str, *, versions: tuple[str,...], planner=None) -> dict:
    if not isinstance(query,str) or not query.strip() or len(query)>4000:
        raise ValueError('invalid query')
    if not versions or any(v not in PADDLEOCR_VERSIONS for v in versions):
        raise ValueError('unknown selected version')
    explicit=explicit_versions(query)
    module=query_module(query)
    intent=retrieval_intent(query)
    # This workspace's default application is the OCR main flow. A colloquial
    # result migration keeps that scope unless a different pipeline is named.
    if module is None and intent['kind']=='result_migration':
        module='ocr'
    requirements=[]
    clauses = _atomic_clauses(query)
    for text,dimension in clauses:
        if not text.strip(): continue
        clause_versions=explicit_versions(text)
        clause_module = query_module(text) or module
        # A follow-up about a pipeline's component option stays on that pipeline.
        # Explicit standalone API names or "独立模块" still open their own scope.
        if (module in ('ocr', 'structure', 'table_recognition_v2', 'doc_preprocessor')
                and clause_module not in ('ocr', 'structure', 'table_recognition_v2', 'doc_preprocessor')
                and not re.search(r'独立|模块', text)
                and not any(re.search(r'\b'+api+r'\s*\(',text) for api in _COMPONENT_APIS)):
            clause_module = module
        dimension_terms={'call':'PaddleOCR def ocr predict input',
                         'parameters':'PaddleOCR def __init__ kwargs lang 参数 默认值',
                         'result':'PaddleOCR ocr_res rec_texts rec_scores 返回'}
        requirements.append({'id':f'r{len(requirements)+1}','query':text.strip(),'fact_dimension':dimension,
                             'retrieval_query':text+' '+dimension_terms.get(dimension,''),
                             'module':clause_module,'symbols':[s for s in _SYMBOL.findall(text)
                                 if s!='PaddleOCR' or re.search(r'\bPaddleOCR\s*\(',text)],
                             'version':next(iter(clause_versions)) if len(clause_versions)==1 else next(iter(explicit)) if len(explicit)==1 else versions[0] if len(versions)==1 else None})
        if len(requirements)==REQUIREMENT_BUDGET:break
    result={'status':'NEEDS_CLARIFICATION' if explicit-set(versions) else 'READY',
            'original_query':query,'versions':list(versions),'language':'zh',
            'namespace':'project_primary','requirements':requirements,'planner_status':'NOT_REQUESTED',
            'omitted_requirements':[text for text,_ in clauses[REQUIREMENT_BUDGET:]],
            'requirement_budget':REQUIREMENT_BUDGET,
            'retrieval_intent':intent}
    if len(explicit)>1 and not explicit-set(versions):
        # Each version is a separate evidence obligation. One high-ranking old
        # guide must not stand in for both sides of an upgrade comparison.
        expanded=[]
        for requirement in requirements:
            for version in sorted(explicit) if requirement['version'] is None else [requirement['version']]:
                expanded.append({**requirement,'id':f'r{len(expanded)+1}','version':version,
                    'version_role': 'baseline' if version == sorted(explicit)[0] else 'target',
                    'evidence_obligation': f'核对 {version} 本版的相关接口、参数或结果事实；'
                        '跨版本解释由两版事实共同支持，不要求单版资料独自说明整个迁移。'})
        result['requirements']=expanded[:REQUIREMENT_BUDGET]
        result['omitted_requirements'] += [f"{r['query']}（{r['version']}）" for r in expanded[REQUIREMENT_BUDGET:]]
    if planner and result['status']=='READY':
        try:
            proposal=planner(query=query,versions=list(versions),requirements=requirements)
            rows=proposal['requirements']
            if not isinstance(rows,list) or not 1<=len(rows)<=4: raise ValueError('plan size')
            for row in rows:
                if not isinstance(row,dict) or not isinstance(row.get('query'),str) or not 0<len(row['query'])<=4000:
                    raise ValueError('plan query')
                if row.get('version') not in versions: raise ValueError('plan version')
                found={'v'+m.group(0).lower().lstrip('v') for m in _VERSION.finditer(row['query'])}
                if found-set(versions): raise ValueError('plan explicit version')
            # Preserve original requirements; proposed queries only augment retrieval, never become facts.
            result['proposed_queries']=[r['query'] for r in rows]
            result['planner_status']='ACCEPTED'
        except Exception:
            result['planner_status']='REJECTED'
    return result
