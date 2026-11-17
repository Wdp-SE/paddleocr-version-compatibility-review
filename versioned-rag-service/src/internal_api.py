"""Internal control plane. Source acquisition, authorization and tasks live server-side."""
import asyncio
import re
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field,model_validator

from src.internal_security import denied_egress,model_scope
from src.internal_tasks import Conflict,fingerprint
from src.internal_workflows import workflow_versions


class Submission(BaseModel):
    model_config=ConfigDict(extra='forbid')
    application_id:str=Field(pattern=r'^[a-z0-9-]{1,64}$')
    head:str=Field(default='HEAD',min_length=1,max_length=160)
    base:str|None=Field(default=None,min_length=1,max_length=160)
    generate_advice:bool=False
    workflow_type:Literal['upgrade_review','configuration_change']='upgrade_review'

    @model_validator(mode='after')
    def supported_configuration(self):
        if self.workflow_type=='configuration_change' and (not self.base or self.generate_advice):
            raise ValueError('Configuration review requires a baseline commit and deterministic review without model advice')
        return self


class ApplicationKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    application_id: str = Field(pattern=r'^[a-z0-9-]{1,64}$')
    head: str = Field(default='HEAD', min_length=1, max_length=160)
    query: str = Field(min_length=1, max_length=4000)
    version: Literal['v2.9.1', 'v3.0.0', 'all', 'auto'] = 'v3.0.0'
    entrypoint: str | None = Field(default=None, min_length=1, max_length=256)
    generate: bool = False


def install_internal(app,*,store,policy,verifier,registry_loader,collector,build_identity,
                     provider,ci_verifier=None,retrieval_ready=None):
    app.state.internal_store=store
    app.state.internal_policy=policy

    def actual_provider():
        value=provider() if callable(provider) else provider
        return value.strip().casefold() if isinstance(value,str) else None

    @asynccontextmanager
    async def generation_scope(request, application):
        lease = None

        async def admit():
            nonlocal lease
            if lease is not None:
                return
            try:
                lease = await asyncio.to_thread(store.reserve_generation, request.state.principal.subject,
                    application, daily_budget=policy.application(application).get('daily_model_requests'))
            except Conflict:
                raise HTTPException(429, 'GENERATION_CAPACITY_EXHAUSTED', headers={'Retry-After': '10'}) from None

        request.state.generation_admission = admit
        token = model_scope.set((policy, application))
        try:
            yield admit
        finally:
            model_scope.reset(token)
            if lease is not None:
                await asyncio.to_thread(store.release_generation, lease)

    @app.middleware('http')
    async def identity(request,call_next):
        if request.method in ('POST','PUT','PATCH'):
            try:
                length=int(request.headers.get('Content-Length','0'))
                if not 0<=length<=500_000:raise ValueError()
            except ValueError:return JSONResponse({'detail':'BODY_BUDGET_EXCEEDED'},status_code=413)
        if request.url.path in ('/health/live','/health/ready'):
            return await call_next(request)
        # CI uses its own trusted issuer; employee JWTs cannot create CI provenance.
        if re.fullmatch(r'/internal/tasks/[a-f0-9]{32}/ci',request.url.path) and request.method=='POST':
            if ci_verifier is None:return JSONResponse({'detail':'TRUSTED_CI_NOT_CONFIGURED'},status_code=503)
            current=ci_verifier
        else:current=verifier
        header=request.headers.get('Authorization','')
        if not header.startswith('Bearer '):
            return JSONResponse({'detail':'AUTHENTICATION_REQUIRED'},status_code=401,headers={'WWW-Authenticate':'Bearer'})
        try:
            request.state.principal=await asyncio.to_thread(current.verify,header[7:])
        except PermissionError:
            return JSONResponse({'detail':'INVALID_IDENTITY'},status_code=401,headers={'WWW-Authenticate':'Bearer'})
        if request.method in ('POST','PUT','PATCH'):
            chunks=[];size=0
            async for chunk in request.stream():
                size+=len(chunk)
                if size>500_000:return JSONResponse({'detail':'BODY_BUDGET_EXCEEDED'},status_code=413)
                chunks.append(chunk)
            request._body=b''.join(chunks)
        if request.url.path.startswith(('/public/','/engineering/')):
            application=request.headers.get('X-Application-ID','')
            subject=request.state.principal.subject
            allowed=set(policy.applications(subject,'read'))|set(policy.applications(subject,'worker'))
            if application not in allowed:return JSONResponse({'detail':'RESOURCE_UNAVAILABLE'},status_code=403)
            if request.method=='POST':
                body=await request.body()
                if len(body)>500_000:return JSONResponse({'detail':'BODY_BUDGET_EXCEEDED'},status_code=413)
                if request.url.path in ('/public/application-search', '/public/application-query'):
                    try:
                        import json
                        if json.loads(body).get('application_id')!=application:raise ValueError()
                    except (ValueError,AttributeError):return JSONResponse({'detail':'APPLICATION_SCOPE_MISMATCH'},status_code=403)
                if request.url.path in ('/public/query','/public/application-query','/public/review-advice','/public/investigation-plan'):
                    reason=denied_egress(policy.application(application),actual_provider(),body.decode('utf8',errors='replace'))
                    if reason:return JSONResponse({'detail':reason,'fallback':'Use read-only retrieval or static review'},status_code=403)
                    async with generation_scope(request, application) as admit:
                        if request.url.path not in ('/public/query','/public/application-query'):
                            try:
                                await admit()
                            except HTTPException as exc:
                                return JSONResponse({'detail': exc.detail}, status_code=exc.status_code, headers=exc.headers)
                        return await call_next(request)
        return await call_next(request)

    def require(request,application,action):
        try:policy.require(request.state.principal.subject,application,action)
        except PermissionError:raise HTTPException(403,'RESOURCE_UNAVAILABLE') from None

    def task_for(request,task):
        try:
            row=store.read(task)
            require(request,row['application'],'read')
            if row['owner']!=request.state.principal.subject:
                policy.require(request.state.principal.subject,row['application'],'review')
            return row
        except (KeyError,PermissionError):raise HTTPException(404,'TASK_UNAVAILABLE') from None

    @app.get('/health/live')
    def live():return {'alive':True}

    @app.get('/health/ready')
    def ready():
        try:
            store.health()
            identity=build_identity()
            keys=('corpus_fingerprint','service_code_fingerprint','retrieval_config_fingerprint','review_code_fingerprint')
            values=[identity.get(k) for k in keys]
            values=[v.get('fingerprint_sha256') if isinstance(v,dict) else v for v in values]
            if any(not re.fullmatch('[a-f0-9]{64}',str(v)) for v in values):raise ValueError()
            if retrieval_ready is None or retrieval_ready() is not True:raise ValueError()
            return {'ready':True,'generation':'optional; governed per application'}
        except Exception:return JSONResponse({'ready':False,'code':'DEPENDENCY_UNAVAILABLE'},status_code=503)

    @app.get('/internal/applications')
    def applications(request:Request):
        allowed=set(policy.applications(request.state.principal.subject,'read'))
        return {'applications':[{'id':row['id'],'name':row['name'],'entrypoints':sorted(row.get('entrypoints', {}))}
                                for row in registry_loader().values() if row['id'] in allowed]}

    @app.post('/internal/application-knowledge')
    async def application_knowledge(payload: ApplicationKnowledgeRequest, request: Request):
        from src import public_api
        from src.application_knowledge import resolve_entrypoint

        application = payload.application_id
        require(request, application, 'read')
        application_policy = policy.application(application)
        if payload.generate:
            reason = denied_egress(application_policy, actual_provider(), payload.query)
            if reason:
                raise HTTPException(403, reason)
        registration = registry_loader().get(application)
        if registration is None:
            raise HTTPException(404, 'APPLICATION_UNAVAILABLE')
        try:
            snapshot = await asyncio.to_thread(collector, registration, payload.head)
            if snapshot['application_id'] != application:
                raise ValueError('Application snapshot mismatch')
            metadata = {'application_id': application, 'application_commit': snapshot['head_commit'],
                        'snapshot_gaps': snapshot.get('gaps', [])}
            version = payload.version
            if version == 'auto' or payload.entrypoint is not None:
                context = resolve_entrypoint(registration, snapshot, payload.entrypoint)
                metadata['entrypoint_context'] = context
                if version == 'auto':
                    version = context['declared_version']
                    if version is None:
                        return {**metadata, 'status': 'NEEDS_ENTRYPOINT_CONTEXT', 'evidence': []}
            query = public_api.ApplicationQueryRequest(files=snapshot['files'], query=payload.query,
                application_id=application, application_commit=snapshot['head_commit'], version=version)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, 'APPLICATION_SNAPSHOT_UNAVAILABLE') from exc
        if not payload.generate:
            result = await public_api.resolve_application_knowledge(query, request, generate=False)
            return {**result, **metadata}

        async with generation_scope(request, application):
            result = await public_api.resolve_application_knowledge(query, request, generate=True)
            return {**result, **metadata}

    @app.post('/internal/tasks',status_code=202)
    def submit(payload:Submission,request:Request):
        require(request,payload.application_id,'submit')
        registration=registry_loader().get(payload.application_id)
        if registration is None:raise HTTPException(404,'APPLICATION_UNAVAILABLE')
        try:
            snapshot=collector(registration,payload.head,base_commit=payload.base)
            baseline=collector(registration,snapshot['base_commit']) if payload.base else None
            if baseline and baseline['head_commit']!=snapshot['base_commit']:
                raise ValueError('Baseline commit identity mismatch')
            policy_hash=fingerprint(policy.value)
            kind,source,target=workflow_versions({'workflow_type':payload.workflow_type})
            body={'snapshot':snapshot,'base_snapshot':baseline,'generate_advice':payload.generate_advice,
                  'workflow_type':kind,'source_version':source,'target_version':target,'build_identity':build_identity(),
                  'access_policy_hash':policy_hash}
            if payload.generate_advice:
                reason=denied_egress(policy.application(payload.application_id),actual_provider(),
                    '\n'.join(f['content'] for s in (snapshot,baseline) if s for f in s['files']))
                if reason:raise HTTPException(403,reason)
            row=store.submit(request.state.principal.subject,payload.application_id,fingerprint(body),body)
            return {'id':row['id'],'state':row['state'],'application_commit':snapshot['head_commit'],
                    'workflow_type':kind,'identity':row['identity'],'runtime_verified':False}
        except Conflict as exc:raise HTTPException(409,str(exc)) from None
        except ValueError as exc:raise HTTPException(422,str(exc)) from None

    @app.get('/internal/tasks/{task}')
    def status(task:str,request:Request):
        row=task_for(request,task)
        value={key:row[key] for key in ('id','application','identity','state','stage','attempts','created','updated','error')}
        if row['state']=='RUNNING' and (row.get('result') or {}).get('partial'):
            value['static_preview']=row['result'].get('static_preview')
        return value

    @app.get('/internal/tasks/{task}/report')
    def report(task:str,request:Request):
        from src.internal_records import regression_coverage
        row=task_for(request,task)
        if row['state']!='DONE':raise HTTPException(409,'REPORT_NOT_COMPLETE')
        records = store.records(task)
        return {'task_id':task,'report':row['result'],'records':records,'events':store.events(task),
                'regression_coverage':regression_coverage(row, records),
                'notice':'Static findings, CI provenance and human decisions are separate; no automatic upgrade approval.'}

    @app.post('/internal/tasks/{task}/cancel')
    def cancel(task:str,request:Request):
        row=task_for(request,task)
        if row['owner']!=request.state.principal.subject:raise HTTPException(403,'OWNER_REQUIRED')
        store.cancel(task,row['owner'])
        return {'id':task,'state':store.read(task)['state']}

    @app.get('/internal/metrics')
    def metrics(request:Request):
        if not policy.applications(request.state.principal.subject,'admin'):raise HTTPException(403,'ADMIN_REQUIRED')
        return store.health()

    @app.post('/internal/tasks/{task}/retry')
    def retry(task:str,request:Request):
        row=task_for(request,task);require(request,row['application'],'submit')
        if row['owner']!=request.state.principal.subject:raise HTTPException(403,'OWNER_REQUIRED')
        try:store.retry(task,row['owner'])
        except Conflict as exc:raise HTTPException(409,str(exc)) from None
        return {'id':task,'state':'QUEUED'}

    @app.post('/internal/tasks/{task}/ci')
    def ci_record(task:str,payload:dict,request:Request):
        from src.internal_records import validate_ci_record
        try:
            row=store.read(task)
            checked=validate_ci_record(row,payload,request.state.principal.claims,policy.application(row['application']))
            store.record(task,'CI',request.state.principal.subject,fingerprint(checked),checked)
            return {'recorded':True,'provenance':'trusted_ci_identity','upgrade_approved':False}
        except KeyError:raise HTTPException(404,'TASK_UNAVAILABLE') from None
        except ValueError:raise HTTPException(422,'CI_BINDING_INVALID') from None

    @app.post('/internal/tasks/{task}/decisions')
    def decision(task:str,payload:dict,request:Request):
        from src.internal_records import validate_decision
        row=task_for(request,task);require(request,row['application'],'review')
        try:
            checked=validate_decision(row,payload,request.state.principal.subject,store.records(task))
            store.record(task,'DECISION',request.state.principal.subject,fingerprint(checked),checked)
            return {'recorded':True,'decision':checked['decision'],'upgrade_approved':False}
        except ValueError:raise HTTPException(422,'DECISION_BINDING_INVALID') from None
