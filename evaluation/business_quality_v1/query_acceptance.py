"""Explicit opt-in live acceptance: official data and registered original app only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
VERSIONS = {'v2.9.1', 'v3.0.0'}


def _has_line_range(row):
    start, end = row.get('line_start'), row.get('line_end')
    return type(start) is int and type(end) is int and 1 <= start <= end


def check_response(value, version, *, mode, snapshot=None):
    """Validate delivery and provenance, leaving semantic grading to reviewers."""
    if not isinstance(value, dict):
        return ['INVALID_RESPONSE']
    key = 'results' if mode in ('search', 'conflict') else 'evidence'
    evidence = value.get(key)
    if mode == 'conflict':
        return [] if value.get('status') == 'NEEDS_CLARIFICATION' and evidence == [] else ['VERSION_CONFLICT_NOT_ENFORCED']
    if mode == 'search' and value.get('status') not in (None, 'OK'):
        return ['SEARCH_FAILED']
    if mode != 'search' and (value.get('status') != 'OK' or not value.get('claims')):
        return ['MISSING_VERIFIED_ANSWER']
    if not isinstance(evidence, list) or not evidence or any(not isinstance(row, dict) for row in evidence):
        return ['NO_VALID_EVIDENCE']
    allowed = VERSIONS if version == 'all' else {version}
    official = [row for row in evidence if row.get('source_identity') != 'submitted_application']
    submitted = [row for row in evidence if row.get('source_identity') == 'submitted_application']
    failures = []
    if {row.get('version') for row in official} != allowed:
        failures.append('UNEXPECTED_OFFICIAL_VERSION_COVERAGE')
    if any(row.get('repository') != 'PaddlePaddle/PaddleOCR' or not row.get('source_id')
           or not row.get('chunk_id') or not _has_line_range(row)
           for row in official):
        failures.append('INVALID_OFFICIAL_PROVENANCE')
    if mode == 'joint':
        snapshot = snapshot or {}
        application_commit = snapshot.get('head_commit')
        if (not application_commit or not submitted or value.get('application_commit') != application_commit
                or any(row.get('application_commit') != application_commit for row in submitted)):
            failures.append('APPLICATION_COMMIT_MISMATCH')
        application_id = snapshot.get('application_id')
        files = {row['path']: row['content'].splitlines() for row in snapshot.get('files', [])}
        hashes = {row['path']: row['sha256'] for row in snapshot.get('inventory', [])}
        if not application_id or value.get('application_id') != application_id or any(
            row.get('application_id') != application_id or not isinstance(row.get('path'), str)
            or row['path'] not in files or not re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256')))
            or row['sha256'] != hashes.get(row['path']) or not _has_line_range(row)
            or row['line_end'] > len(files[row['path']])
            or row.get('content') != '\n'.join(files[row['path']][row['line_start']-1:row['line_end']])
            for row in submitted
        ):
            failures.append('INVALID_APPLICATION_PROVENANCE')
    elif submitted:
        failures.append('UNEXPECTED_APPLICATION_EVIDENCE')
    if mode != 'search':
        by_id = {row['chunk_id']: row for row in evidence if isinstance(row.get('chunk_id'), str)}
        claims = value['claims']
        sources = value.get('sources')
        cited = set()
        if (not isinstance(claims, list) or not all(isinstance(claim, dict) for claim in claims)
                or not isinstance(sources, list)
                or any(not isinstance(row, dict) or not isinstance(row.get('chunk_id'), str)
                       or by_id.get(row['chunk_id']) != row for row in sources)):
            return [*failures, 'INVALID_CITATIONS']
        for claim in claims:
            refs = claim.get('evidence_ids')
            if (not isinstance(claim.get('text'), str) or not claim['text'].strip()
                    or not isinstance(refs, list) or not refs
                    or any(not isinstance(ref, str) or ref not in by_id for ref in refs)):
                return [*failures, 'INVALID_CITATIONS']
            cited.update(refs)
        if cited != {row['chunk_id'] for row in sources}:
            failures.append('INVALID_CITATIONS')
        if mode == 'joint' and not (
            cited.intersection(row.get('chunk_id') for row in official)
            and cited.intersection(row.get('chunk_id') for row in submitted)
        ):
            failures.append('JOINT_CITATION_SCOPE_MISSING')
    return failures


def run(url, output, *, generate=False):
    output=Path(output)
    if output.exists():raise ValueError('Use a new output path; never overwrite a previous run')
    output.parent.mkdir(parents=True,exist_ok=True)
    def post(path,payload):
        request=urllib.request.Request(url.rstrip('/')+path, data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json','X-Demo-Session-ID':session},method='POST')
        with urllib.request.urlopen(request,timeout=600) as response:return json.load(response)
    session=uuid.uuid4().hex
    with urllib.request.urlopen(url.rstrip('/')+'/public/workspace',timeout=45) as response:workspace=json.load(response)
    rows=[]
    result={'schema_version':1,'origin':'official PaddleOCR and registered student original application',
        'generation_requested':generate,'human_accuracy':None,
        'assessment':'developer_authored_acceptance_not_independent_blind',
        'workspace':{k:workspace.get(k) for k in ('workspace_id','corpus_fingerprint','service_code_fingerprint','retrieval_config_fingerprint')},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
    cases=[('single', 'OCR 3.0.0 的 rec_texts 和 rec_scores 分别表示什么？','v3.0.0'),
           ('single_repeat','OCR 3.0.0 的 rec_texts 和 rec_scores 分别表示什么？','v3.0.0'),
           ('compound','PaddleOCR 2.9.1 升级到 3.0.0，调用入口、初始化参数和结果读取应分别核查什么？','all'),
           ('colloquial_compound','OCR 从 2.9.1 迁到 3.0.0，构造器、如何调用、返回结构分别要检查哪些内容？请附上两版出处。','all'),
           ('legacy_method','OCR 3.0.0 的 ocr() 方法还保留吗，它与 predict() 是什么关系？','v3.0.0'),
           ('default_omission','OCR 2.9.1 调用 ocr() 时省略 rec 参数，会直接返回 cls_res 吗？请核对默认值和返回分支。','v2.9.1')]
    cases.append(('version_conflict', '请解释 PaddleOCR 3.0.0 的 rec_texts。', 'v2.9.1'))
    if generate:
        sys.path.insert(0,str(ROOT/'demo-ui'))
        from services.registered_application import load_registry, collect_snapshot
        registry=load_registry()
        app=next(value for value in registry.values() if value['source_identity']=='student_poc_application')
        snapshot=collect_snapshot(app,'HEAD')
        cases.append(('joint','OCR 3.0.0 的 rec_texts 和 rec_scores 如何对应本应用的 text、confidence 与 page_index 输出要求？','v3.0.0'))
    for name,question,version in cases:
        started=time.perf_counter()
        payload={'query':question,'version':version,'language':'zh','top_k':5,'retrieval_policy':'paddleocr_evidence'}
        endpoint='/public/query' if generate else '/public/search'
        mode = 'generate' if generate else 'search'
        if name == 'version_conflict':
            endpoint, mode = '/public/search', 'conflict'
        if name=='joint':
            endpoint, mode = '/public/application-query', 'joint'
            payload={'files':snapshot['files'],'query':question,'version':version,'top_k':5,
                     'application_id':snapshot['application_id'],'application_commit':snapshot['head_commit']}
        try:
            value=post(endpoint,payload)
            failures = check_response(value, version, mode=mode, snapshot=snapshot if mode == 'joint' else None)
            rows.append({'id':name,'query':question,'version':version,'seconds':round(time.perf_counter()-started,3),
                         'status': 'FAILED' if failures else 'PASSED', 'failures': failures,
                         'response':value,'human_verdict':'NOT_REVIEWED'})
        except Exception as exc:
            rows.append({'id':name,'query':question,'version':version,'seconds':round(time.perf_counter()-started,3),
                         'status':'REQUEST_FAILED','failure_type':type(exc).__name__,'human_verdict':'NOT_REVIEWED'})
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({key: rows[-1][key] for key in ('id', 'seconds', 'status')}, ensure_ascii=True), flush=True)
    result['passed'] = all(row['status'] == 'PASSED' for row in rows)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8776')
    parser.add_argument('--output',required=True)
    parser.add_argument('--generate',action='store_true',help='Explicitly call configured provider using only public/original demo data')
    args=parser.parse_args(argv)
    return 0 if run(args.url, args.output, generate=args.generate)['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
