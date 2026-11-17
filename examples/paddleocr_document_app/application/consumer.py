"""Downstream contract independent of OCR vendor result shape."""
import json
import math


def to_json(lines, *, page_count=None):
    if page_count is not None and (type(page_count) is not int or not 0<=page_count<=10000):
        raise ValueError('invalid page count')
    pages={} if page_count is None else {i:[] for i in range(page_count)}
    for line in lines:
        page=line['page_index']; text=line['text']; confidence=line['confidence']
        if type(page) is not int or page<0 or not isinstance(text,str) or page_count is not None and page>=page_count:
            raise ValueError('invalid normalized line')
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('invalid OCR confidence')
        pages.setdefault(page,[]).append(text)
    result={'schema_version':1,'pages':[{'page_index':i,'text':'\n'.join(pages[i])} for i in sorted(pages)]}
    json.dumps(result,ensure_ascii=False,allow_nan=False)
    return result
