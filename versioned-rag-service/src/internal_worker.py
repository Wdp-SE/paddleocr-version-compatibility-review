"""Independent leased worker; never imports or executes submitted application code."""
import argparse
import logging
import os
import re
import hashlib
from pathlib import Path
import signal
import sys
from threading import Event,Thread
import uuid

from src.internal_tasks import TaskStore,fingerprint
from src.internal_workflows import workflow_versions

logger=logging.getLogger(__name__)


class TokenFileAuth:
    """Read an atomically refreshed service token at each HTTP request, not per job."""
    def __init__(self,path,*,before_request=None):
        self.path=Path(path)
        self.before_request=before_request

    def __call__(self,request):
        if self.before_request:self.before_request()
        token=self.path.read_text(encoding='utf8').strip()
        if not re.fullmatch(r'[A-Za-z0-9._~-]{1,16000}',token):
            raise ValueError('Worker service token unavailable or malformed')
        request.headers['Authorization']='Bearer '+token
        return request


def run_one(store,worker,execute,*,heartbeat_seconds=10):
    claim=store.claim(worker)
    if claim is None:return False
    stopped=Event();lost=Event()
    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            try:
                if not store.heartbeat(claim):lost.set();return
            except Exception:lost.set();return
    def cancelled():
        if lost.is_set():return True
        try:
            current=store.read(claim['id'])
            return (current['state']!='RUNNING' or current['lease_token']!=claim['lease_token']
                or current['lease_epoch']!=claim['lease_epoch'] or current['lease_until']<=store.clock())
        except Exception:return True
    def progress(value):
        stage=value.get('stage') if isinstance(value,dict) else value
        if stage in store.stages and not store.heartbeat(claim,stage=stage):lost.set()
        if isinstance(value,dict) and value.get('static_preview'):
            preview=value['static_preview']
            files=(claim['request'].get('snapshot') or {}).get('files',[])
            expected=sorted((f['path'],hashlib.sha256(f['content'].encode()).hexdigest()) for f in files)
            observed=sorted((f.get('path'),f.get('sha256')) for f in preview.get('files',[]))
            if expected and expected==observed and not store.save_preview(claim,preview):lost.set()
    # Preserve the executor callable contract used by deployments and tests.
    # Only author-owned executor code consumes this internal runtime attachment.
    from src.graph_runtime import TaskGraphRuntime
    progress.graph_runtime=TaskGraphRuntime(store,claim,cancelled=cancelled)
    thread=Thread(target=heartbeat,name='review-lease',daemon=True);thread.start()
    try:
        result=execute(claim,cancelled,progress)
        if not cancelled():store.finish(claim,result)
    except Exception as exc:
        logger.warning('review_worker task_id=%s error_code=WORKER_EXECUTION_FAILED',claim['id'])
        from src.graph_runtime import UnsafeGraphReplay
        code='GRAPH_REPLAY_UNSAFE' if isinstance(exc,UnsafeGraphReplay) else 'WORKER_EXECUTION_FAILED'
        store.finish(claim,{'runtime_verified':False},error=code)
    finally:
        stopped.set();thread.join(timeout=heartbeat_seconds+1)
    return True


def configured_executor(config,*,base_url,token_file):
    from src.internal_security import AccessPolicy,denied_egress
    import json
    root=Path(__file__).resolve().parents[2]
    from src.build_identity import review_execution_fingerprint
    # Keep the Agent package ahead of demo-ui/app.py; these are author-owned service modules.
    sys.path[:0]=[str(root/'change-review-agent'),str(root/'demo-ui')]
    from app.paddleocr_review import PaddleOCRReviewAgent
    from services.public_knowledge_client import PublicKnowledgeClient
    from services.registered_review import run_registered_review
    from services.configuration_workflow import run_configuration_review
    import requests

    def execute(row,cancelled,progress):
        policy=AccessPolicy(json.loads(config.policy_path.read_text(encoding='utf-8-sig')))
        policy.require(row['owner'],row['application'],'submit')
        body=row['request']
        kind,source_version,target_version=workflow_versions(body)
        if kind=='configuration_change' and (not body.get('base_snapshot') or body.get('generate_advice')):
            raise ValueError('Configuration review requires paired commits without model advice')
        if body['build_identity'].get('review_code_fingerprint')!=review_execution_fingerprint(root):
            raise ValueError('Worker analysis build changed; resubmit with matching deployment')
        if body['access_policy_hash']!=fingerprint(policy.value):raise ValueError('Access policy changed')
        def still_authorized():
            if cancelled():raise ValueError('Review cancelled or lease lost')
            current_policy=AccessPolicy(json.loads(config.policy_path.read_text(encoding='utf-8-sig')))
            current_policy.require(row['owner'],row['application'],'submit')
            if fingerprint(current_policy.value)!=body['access_policy_hash']:
                raise PermissionError('Access policy changed during review')
        runtime=getattr(progress,'graph_runtime',None)
        if runtime is not None:runtime.authorize=still_authorized
        session=requests.Session()
        session.auth=TokenFileAuth(token_file,before_request=still_authorized)
        session.headers.update({'X-Application-ID':row['application']})
        client=PublicKnowledgeClient(base_url,session=session,timeout=30,retry_limit=0)
        try:
            workspace=client.workspace()
            current={k:workspace.get(k) for k in ('corpus_fingerprint','service_code_fingerprint','retrieval_config_fingerprint','review_code_fingerprint')}
            if current!=body['build_identity']:raise ValueError('Build or corpus changed; resubmit')
            if hasattr(client,'bind_service_identity'):client.bind_service_identity(workspace)
            if body['generate_advice']:
                content='\n'.join(f['content'] for snap in (body['snapshot'],body['base_snapshot']) if snap for f in snap['files'])
                if denied_egress(policy.application(row['application']),workspace.get('generation_provider'),content):
                    raise PermissionError('Egress denied')
            # run_registered_review evaluates baseline before candidate. Explicit
            # scopes prevent two snapshots from sharing one graph checkpoint.
            review_passes=iter(([('baseline',body['base_snapshot'])] if body.get('base_snapshot') else [])
                               + [('candidate',body['snapshot'])])
            def analyze(gateway,files,**kwargs):
                scope,expected=next(review_passes)
                if fingerprint(files)!=fingerprint(expected['files']):
                    raise ValueError('Review pass does not match submitted snapshot')
                pass_runtime=runtime.scoped(scope) if runtime is not None else None
                return PaddleOCRReviewAgent(gateway).analyze(files,cancelled=cancelled,on_progress=progress,
                                                            investigation_progress=progress,graph_runtime=pass_runtime,**kwargs)
            still_authorized()
            if kind=='configuration_change':
                progress('static_review')
                result=run_configuration_review(client,body['base_snapshot']['files'],body['snapshot']['files'],
                    snapshot=body['snapshot'],base_snapshot=body['base_snapshot'],graph_runtime=runtime)
                progress('evidence_validation')
            else:
                result=run_registered_review(client,body['snapshot'],base_snapshot=body['base_snapshot'],
                    analyze=analyze,context={'application_id':row['application'],'purpose':'internal_upgrade_preflight'},
                    source_version=source_version,target_version=target_version,
                    generate_advice=body['generate_advice'])
            result['workflow_type']=kind
            still_authorized()
            final_workspace=client.workspace()
            if {k:final_workspace.get(k) for k in current}!=body['build_identity']:
                raise ValueError('Build or corpus changed during review')
            result['internal_task']={'id':row['id'],'owner':row['owner'],'identity':row['identity']}
            return result
        finally:session.close()
    return execute


def main():
    from src.internal_security import InternalConfig
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    config=InternalConfig.from_env(os.environ)
    base=os.environ.get('INTERNAL_API_URL','').rstrip('/')
    from urllib.parse import urlsplit
    parsed=urlsplit(base)
    if parsed.scheme!='https' and not (config.development and parsed.hostname in ('127.0.0.1','localhost','api')):
        raise ValueError('Worker requires configured trusted HTTPS internal API')
    token_file=os.environ.get('INTERNAL_WORKER_TOKEN_FILE','')
    if not token_file:raise ValueError('Worker requires refreshed service-identity token file')
    store=TaskStore(config.database_url,initialize=config.development)
    execute=configured_executor(config,base_url=base,token_file=token_file)
    stopping=Event()
    for name in ('SIGINT','SIGTERM'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),lambda *_:stopping.set())
    worker='review-'+uuid.uuid4().hex
    while not stopping.is_set():
        worked=run_one(store,worker,execute)
        if args.once:break
        if not worked:stopping.wait(1)


if __name__=='__main__':main()
