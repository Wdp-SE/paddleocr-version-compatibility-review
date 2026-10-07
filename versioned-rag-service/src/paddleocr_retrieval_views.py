"""Bounded retrieval windows; citations always point into unchanged source blocks."""
from __future__ import annotations
import hashlib
import re
from collections import OrderedDict


def module_for(chunk: dict) -> str:
    path = chunk.get('document_path', '').lower()
    # Historical guides predate the pipeline_usage/ directory. Their explicit
    # source role must survive windows that don't repeat the imported class.
    if path in ('doc/doc_ch/whl.md','doc/doc_ch/inference_ppocr.md','doc/doc_ch/inference_args.md'):
        return 'ocr'
    if path == 'tests/test_paddleocr_api.py':
        text = chunk.get('content','')
        if re.search(r'\bdef\s+(?:test_ocr\w*|ocr_engine)\b',text):return 'ocr'
        if re.search(r'\bdef\s+(?:test_structure\w*|structure_engine)\b',text):return 'structure'
    # A pipeline's test body does not necessarily repeat its constructor. Its
    # path still owns the scope, including assertion-only source blocks.
    if path == 'tests/pipelines/test_ocr.py':
        return 'ocr'
    if 'module_usage/' in path:
        return path.rsplit('/', 1)[-1].split('.')[0]
    if 'pipeline_usage/' in path:
        name=path.rsplit('/',1)[-1].split('.')[0]
        return {'ocr':'ocr','pp-structurev3':'structure','doc_preprocessor':'doc_preprocessor',
                'table_recognition_v2':'table_recognition_v2'}.get(name,'general')
    if 'preprocessor' in path: return 'doc_preprocessor'
    if 'structure' in path: return 'structure'
    if path.endswith('/ocr.md') or path.endswith('/ocr.py') or path == 'paddleocr.py': return 'ocr'
    if 'common_args' in path: return 'shared'
    return 'general'


def _context(parent: dict) -> str:
    pieces = [parent.get('document_title',''), parent.get('heading','')]
    lines = parent['content'].splitlines()
    if len(lines)>1 and lines[0].lstrip().startswith('|') and '---' in lines[1]:
        pieces.append('\n'.join(lines[:2]))
    return '\n'.join(p for p in pieces if p)


def _span_text(parent: dict, span: dict) -> str:
    text = parent['content']; lines = text.splitlines()
    start, end = span.get('line_start'), span.get('line_end')
    base = parent['line_start']
    if (type(start) is not int or type(end) is not int or start<base or end<start
            or end>base+len(lines)-1):
        raise ValueError('evidence span lines outside canonical parent')
    if 'char_start' in span or 'char_end' in span:
        lo, hi=span.get('char_start'),span.get('char_end')
        if type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=len(text):
            raise ValueError('invalid evidence span character offsets')
        if base+text[:lo].count('\n')!=start or base+text[:hi-1].count('\n')!=end:
            raise ValueError('evidence span line/character mismatch')
        value=text[lo:hi]
    else:
        value='\n'.join(lines[start-base:end-base+1])
    if 'content' in span and span['content']!=value:
        raise ValueError('evidence span content differs from canonical source')
    return value


def build_views(chunks: list[dict], *, token_count, max_tokens: int = 384) -> list[dict]:
    if type(max_tokens) is not int or max_tokens<16:
        raise ValueError('invalid window token budget')
    views=[]
    for parent in chunks:
        text=parent['content']
        context=_context(parent)
        while context and token_count(context+'\n')>max_tokens//4:
            context=context[:-1]
        prefix=context+'\n' if context else ''
        lo=0
        while lo<len(text):
            # Binary search actual token budget, including inherited context.
            left,right=lo+1,len(text)
            best=lo
            while left<=right:
                mid=(left+right)//2
                if token_count(prefix+text[lo:mid])<=max_tokens: best=mid; left=mid+1
                else: right=mid-1
            if best==lo: raise ValueError('one character exceeds tokenizer budget')
            hi=best
            if hi<len(text):
                boundary=text.rfind('\n',lo,hi)
                if boundary>lo: hi=boundary+1
            # A trailing newline belongs to the preceding line, not a nonexistent next line.
            span={'line_start':parent['line_start']+text[:lo].count('\n'),
                  'line_end':parent['line_start']+text[:hi-1].count('\n'),
                  'char_start':lo,'char_end':hi,'content':text[lo:hi]}
            identity={k:parent[k] for k in ('source_id','version','source_sha256') if k in parent}
            views.append({**identity,**span,'parent_chunk_id':parent['chunk_id'],
                          'view_id':hashlib.sha256(f"{parent['chunk_id']}:{lo}:{hi}".encode()).hexdigest(),
                          'module':module_for(parent),'context_char_end':len(context),
                          'retrieval_text':prefix+span['content'], 'token_count':token_count(prefix+span['content'])})
            lo=hi
    return views


def validate_views(chunks: list[dict], views: list[dict]) -> None:
    parents={c['chunk_id']:c for c in chunks}; seen=set()
    for view in views:
        parent=parents.get(view.get('parent_chunk_id'))
        if parent is None or any(view.get(k)!=parent.get(k) for k in ('source_id','version','source_sha256')):
            raise ValueError('retrieval view identity mismatch')
        value=_span_text(parent,view)
        count=view.get('context_char_end')
        context=_context(parent)
        if type(count) is not int or not 0<=count<=len(context): raise ValueError('invalid view context')
        expected=(context[:count]+'\n' if count else '')+value
        vid=hashlib.sha256(f"{parent['chunk_id']}:{view['char_start']}:{view['char_end']}".encode()).hexdigest()
        if (view.get('retrieval_text')!=expected or view.get('module')!=module_for(parent)
                or view.get('view_id')!=vid or vid in seen):
            raise ValueError('retrieval view identity or derived text mismatch')
        seen.add(vid)


def materialize_evidence(chunks: list[dict], selected: list[dict]) -> list[dict]:
    parents={c['chunk_id']:c for c in chunks}; output=OrderedDict()
    for view in selected:
        parent=parents.get(view.get('parent_chunk_id'))
        if parent is None: raise ValueError('unknown evidence parent')
        if any(k in view and view[k]!=parent.get(k) for k in ('source_id','version','source_sha256')):
            raise ValueError('evidence identity mismatch')
        span={k:view[k] for k in ('line_start','line_end','char_start','char_end') if k in view}
        span['content']=_span_text(parent,view)
        row=output.setdefault(parent['chunk_id'],{**parent,'module':module_for(parent),'evidence_spans':[]})
        if span not in row['evidence_spans']: row['evidence_spans'].append(span)
    return list(output.values())


def evidence_text(hit: dict) -> str:
    """Exact retrieval spans, used by retrieval metrics and provenance checks."""
    spans = hit.get('evidence_spans')
    if spans is None: return hit['content']
    if not isinstance(spans, list) or not spans: raise ValueError('empty evidence spans')
    return '\n'.join(_span_text(hit, span) for span in spans)


def generation_evidence_text(hit: dict) -> str:
    spans=hit.get('evidence_spans')
    if spans is None: return hit['content']
    if not isinstance(spans,list) or not spans: raise ValueError('empty evidence spans')
    # Retrieve short windows, then reconstruct bounded source units for the
    # answer/checker. Added context is copied from the same canonical block only.
    text = hit['content']; lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    ranges = []
    for span in spans:
        _span_text(hit, span)  # Reject forged spans before expanding anything.
        start = span['line_start'] - hit['line_start']
        end = span['line_end'] - hit['line_start'] + 1
        lo, hi = max(0, start-2), min(len(lines), end+2)
        # PaddleOCR guides use HTML tables as well as Markdown tables.
        char_lo = span.get('char_start', offsets[start])
        char_hi = span.get('char_end', offsets[end])
        table_start = text.rfind('<table', 0, char_hi)
        table_end = text.find('</table>', char_hi)
        if table_start >= 0 and table_end >= 0 and text.rfind('</table>', 0, char_lo) < table_start:
            header = re.search(r'<thead\b.*?</thead>', text[table_start:table_end], re.S)
            if header:
                ranges.append((table_start+header.start(), table_start+header.end()))
            row_start = text.rfind('<tr', table_start, char_lo+1)
            row_end = text.find('</tr>', char_hi)
            if row_start >= 0 and row_end >= char_hi and row_end+5-row_start <= 6000:
                ranges.append((row_start, row_end+5))
        # Preserve a Markdown table's header even when the hit is a later row.
        if any(lines[i].lstrip().startswith('|') for i in range(start, end)):
            header = start
            while header > 0 and lines[header-1].lstrip().startswith('|'): header -= 1
            if header+1 < len(lines) and '---' in lines[header+1]:
                if offsets[hi]-offsets[header] <= 6000: lo = min(lo, header)
                else:
                    ranges.append((offsets[header], offsets[header+2]))
        # A complete fenced example is more useful than an isolated code line.
        fences = [i for i in range(start) if lines[i].lstrip().startswith('```')]
        if len(fences) % 2:
            close = next((i for i in range(end, len(lines)) if lines[i].lstrip().startswith('```')), None)
            if close is not None and offsets[close+1]-offsets[fences[-1]] <= 6000:
                lo, hi = fences[-1], close+1
        if offsets[hi]-offsets[lo] > 6000:
            ranges.append((char_lo,char_hi))
            continue
        ranges.append((offsets[lo], offsets[hi]))
    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1]: merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else: merged.append((lo, hi))
    value='\n'.join(text[lo:hi] for lo, hi in merged)
    # Exact selected windows take precedence over optional surrounding context.
    if len(value)>6000: value=evidence_text(hit)
    return value[:6000]
