"""Original downstream contract adapter; this module does not execute review inputs."""
import math
import numpy as np


def _row(text, score, page):
    if not isinstance(text,str) or isinstance(score,bool):raise ValueError('invalid OCR text or score')
    try:score=float(score)
    except (ValueError,TypeError):raise ValueError('invalid OCR score') from None
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('invalid OCR score')
    return {'text':text,'confidence':score,'page_index':page}


def normalize_v2(result):
    if not isinstance(result,list):raise ValueError('expected v2 page list')
    rows=[]
    for page_index,page in enumerate(result):
        if page is None:continue
        if not isinstance(page,list):raise ValueError('invalid v2 page')
        for line in page:
            if not isinstance(line,(list,tuple)) or len(line)!=2:raise ValueError('invalid v2 line')
            text_score=line[1]
            if not isinstance(text_score,(list,tuple)) or len(text_score)!=2:raise ValueError('invalid v2 text/score')
            rows.append(_row(text_score[0],text_score[1],page_index))
    return rows


def normalize_v3(result):
    rows=[]
    for page_index,page in enumerate(result):
        if not isinstance(page,dict) or 'rec_texts' not in page or 'rec_scores' not in page:
            raise ValueError('expected v3 rec_texts/rec_scores')
        texts,scores=page['rec_texts'],page['rec_scores']
        if not isinstance(texts,(list,tuple)) or not (isinstance(scores,(list,tuple)) or isinstance(scores,np.ndarray) and scores.ndim==1):
            raise ValueError('expected v3 text/score arrays')
        if len(texts)!=len(scores):raise ValueError('v3 text/score length mismatch')
        rows.extend(_row(text,score,page_index) for text,score in zip(texts,scores))
    return rows
