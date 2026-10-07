"""Bounded text inputs and original fixture loading, never user file execution."""
import hashlib
import json
import re
from pathlib import Path


def parse_application_inputs(rows:list[dict])->list[dict]:
    if not isinstance(rows,list) or not 1<=len(rows)<=12:raise ValueError('需要 1–12 个应用文件')
    total=0;seen=set();output=[]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'path','content'}:raise ValueError('每个文件仅包含路径和文本')
        path,text=row['path'],row['content']
        if (not isinstance(path,str) or not 1<=len(path)<=160 or path!=path.strip() or '\\' in path or ':' in path
                or path.startswith('/') or any(ord(c)<32 for c in path)
                or any(p in ('','.','..') or p.endswith((' ','.')) or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)',p,re.I) for p in path.split('/'))):
            raise ValueError('文件路径必须是安全的相对标签')
        if path.casefold() in seen:raise ValueError('duplicate application path')
        seen.add(path.casefold())
        if not isinstance(text,str) or not text:raise ValueError('文件内容不能为空')
        try:raw=text.encode('utf-8')
        except UnicodeEncodeError:raise ValueError('需要有效 UTF-8 文本') from None
        total+=len(raw)
        if len(raw)>50000 or total>200000 or len(text.splitlines())>5000:raise ValueError('文件文本超出审查预算')
        output.append({'path':path,'content':text})
    return output


def application_fingerprint(files,versions,settings,corpus_fingerprint)->str:
    return hashlib.sha256(json.dumps({'files':parse_application_inputs(files),'versions':versions,
        'settings':settings,'corpus':corpus_fingerprint},ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def load_original_application()->list[dict]:
    root=Path(__file__).resolve().parents[2]/'examples/paddleocr_document_app'
    names=('ocr_client.py','normalizer.py','consumer.py','pipeline.py','contract.json','requirements-v2.txt',
           'design.md','upgrade.md','tests/test_contract.py')
    manifest=json.loads((root/'application/application_manifest.json').read_text(encoding='utf-8'))
    registry={r['path']:r['sha256'] for r in manifest['files']}
    rows=[]
    for name in names:
        raw=(root/'application'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=registry.get(name):raise ValueError('原创应用资料哈希不匹配')
        rows.append({'path':'application/'+name,'content':raw.decode('utf-8')})
    rows.append({'path':'result_adapter.py','content':(root/'result_adapter.py').read_text(encoding='utf-8')})
    return parse_application_inputs(rows)
