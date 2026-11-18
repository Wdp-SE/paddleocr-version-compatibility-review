"""Derived example scope from explicit neighbouring source text; never rewrites sources."""
from __future__ import annotations
import re
from collections import defaultdict


def example_applicability(chunks):
    documents=defaultdict(list)
    for row in chunks:
        documents[(row.get('source_id'),row.get('document_path'),row.get('version'))].append(row)
    result={}
    for rows in documents.values():
        active=None; heading=None
        for row in sorted(rows,key=lambda r:r.get('line_start',0)):
            text=row['content']
            if row.get('heading')!=heading or re.match(r'^#{1,6}\s',text):active=None
            heading=row.get('heading')
            # Only an explicit prose introduction scopes the following fenced
            # example. A title mentioning two versions does not select either.
            marker=re.search(r'(?:以下是|在)\s*PaddleOCR\s+(\d+)\.[xX]\s*[^\n]*(?:使用方式|示例如下)',text)
            if marker:
                family=marker.group(1)+'.x'
                release=str(row.get('version','')).lstrip('v').split('.')[0]
                active={'described_version_family':family,
                        'role':'target_example' if marker.group(1)==release else 'baseline_example',
                        'basis':{'chunk_id':row['chunk_id'],'line':row['line_start']+text[:marker.start()].count('\n'),
                                 'text':marker.group(0)}}
            if active and (marker or text.lstrip().startswith('```')):
                result[row['chunk_id']]={**active,'publication_version':row.get('version')}
            elif text.strip() and not marker:
                active=None
    return result


def version_applies(row,version):
    family=(row.get('applicability') or {}).get('described_version_family')
    if family:
        return row.get('version')==version and str(version).lstrip('v').split('.')[0]==family.split('.')[0]
    return not version or row.get('version')==version


def claim_scope_conflict(text,rows):
    majors=set(re.findall(r'(?<![\w.])v?(\d+)\.\d+\.\d+(?![\w.])',text,re.I))
    return bool(majors) and all((r.get('applicability') or {}).get('described_version_family')
        and (r['applicability']['described_version_family'].split('.')[0] not in majors) for r in rows)


def applicability_note(row):
    scope=row.get('applicability')
    if not scope:return ''
    return ('片段适用性（由原文引导语推导，非新增事实）：文档发布版本 '+str(row.get('version'))+
            '；片段描述 '+scope['described_version_family']+'；角色 '+scope['role']+
            '；原文引导语 '+scope['basis']['text']+'。不得将旧版示例当作新版契约。')
