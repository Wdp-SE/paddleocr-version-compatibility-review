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

def explicit_versions(text: str) -> set[str]:
    return {'v'+m.group(0).lower().lstrip('v') for m in _VERSION.finditer(text)}


def query_module(text: str) -> str | None:
    # A named pipeline owns its component options; mentioning the component in
    # that question does not silently switch the API scope to a standalone module.
    for pattern, module in (
        (r'PPStructure(?:V3)?', 'structure'),
        (r'(?<![\w-])OCR(?!\w)|\bPaddleOCR\s*\(', 'ocr'),
        (r'文档预处理|DocPreprocessor', 'doc_preprocessor'),
        (r'表格识别产线|通用表格识别v2|TableRecognitionPipelineV2', 'table_recognition_v2'),
    ):
        if re.search(pattern, text): return module
    values={v for k,v in _MODULES.items() if k in text}
    # The product name is not a pipeline. Only an explicit API call or standalone
    # OCR denotes the OCR pipeline; PP-OCR model families also name the product.
    if re.search(r'(?<![\w-])OCR(?!\w)|\bPaddleOCR\s*\(', text):
        values.add('ocr')
    if len(values)>1:values.discard('ocr')
    return next(iter(values)) if len(values)==1 else None


def claim_scope_module(text: str, requirements: list[dict]) -> str | None:
    """Descriptions of result fields are not switches to standalone submodules."""
    explicit = re.search(r'PPStructure(?:V3)?|\bPaddleOCR\s*\(|(?<![\w-])OCR(?!\w)|文档预处理|'
                         r'(?:文本检测|文本识别|方向分类|图像矫正|表格结构|版面检测)模块|'
                         r'^\s*(?:文本检测|文本识别|方向分类|图像矫正|表格结构|版面检测)', text)
    if explicit: return query_module(text)
    modules={r['module'] for r in requirements if r.get('module')}
    return next(iter(modules)) if len(modules)==1 else None


def plan_query(query: str, *, versions: tuple[str,...], planner=None) -> dict:
    if not isinstance(query,str) or not query.strip() or len(query)>4000:
        raise ValueError('invalid query')
    if not versions or any(v not in PADDLEOCR_VERSIONS for v in versions):
        raise ValueError('unknown selected version')
    explicit=explicit_versions(query)
    module=query_module(query)
    requirements=[]
    for text in re.split(r'[；;\n？?]+|以及',query):
        if not text.strip(): continue
        clause_versions=explicit_versions(text)
        requirements.append({'id':f'r{len(requirements)+1}','query':text.strip(),
                             'module':query_module(text) or module,'symbols':[s for s in _SYMBOL.findall(text)
                                 if s!='PaddleOCR' or re.search(r'\bPaddleOCR\s*\(',text)],
                             'version':next(iter(clause_versions)) if len(clause_versions)==1 else next(iter(explicit)) if len(explicit)==1 else versions[0] if len(versions)==1 else None})
        if len(requirements)==4:break
    result={'status':'NEEDS_CLARIFICATION' if explicit-set(versions) else 'READY',
            'original_query':query,'versions':list(versions),'language':'zh',
            'namespace':'project_primary','requirements':requirements,'planner_status':'NOT_REQUESTED'}
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
