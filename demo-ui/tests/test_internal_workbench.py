from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_internal_page_requires_proxy_identity_and_does_not_start_anonymous_job(monkeypatch):
    monkeypatch.setenv('APP_ENV','internal')
    monkeypatch.setattr('internal_workbench.identity_header',lambda:None)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py').run()
    assert not app.exception
    assert any('企业身份' in x.value for x in app.error)
    assert not any(b.label=='提交升级审查' for b in app.button)


def test_authenticated_page_submits_once_and_navigation_only_reads_status(monkeypatch):
    calls=[];task='a'*32
    monkeypatch.setenv('APP_ENV','internal')
    monkeypatch.setattr('internal_workbench.identity_header',lambda:'Bearer test-only')
    def request(self,method,path,**kwargs):
        calls.append((method,path))
        if path=='/internal/applications':return {'applications':[{'id':'document-ocr','name':'扫描应用'}]}
        if method=='POST' and path=='/internal/tasks':return {'id':task}
        if path=='/internal/tasks/'+task:return {'state':'QUEUED','stage':'queued','attempts':0}
        raise AssertionError((method,path))
    monkeypatch.setattr('services.internal_client.InternalClient._request',request)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py',default_timeout=10).run()
    app.radio[0].set_value('变更审查').run()
    next(b for b in app.button if b.label=='提交升级审查').click().run()
    assert not app.exception
    app.run();app.run()
    assert calls.count(('POST','/internal/tasks'))==1
    assert calls.count(('GET','/internal/tasks/'+task))>=1


def test_terminal_task_stops_background_polling(monkeypatch):
    monkeypatch.setenv('APP_ENV','internal')
    monkeypatch.setattr('internal_workbench.identity_header',lambda:'Bearer test-only')
    def request(self,method,path,**kwargs):
        if path=='/internal/applications':return {'applications':[{'id':'document-ocr','name':'扫描应用'}]}
        return {'state':'CANCELLED','stage':'completed','attempts':1,'error':None}
    monkeypatch.setattr('services.internal_client.InternalClient._request',request)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py').run()
    app.radio[0].set_value('变更审查').run()
    next(x for x in app.text_input if x.label=='任务编号').set_value('a'*32).run()
    assert not app.exception
    assert app.session_state['internal_polling'] is None


def test_changing_version_does_not_display_prior_version_answer(monkeypatch):
    monkeypatch.setenv('APP_ENV','internal')
    monkeypatch.setattr('internal_workbench.identity_header',lambda:'Bearer test-only')
    calls=[]
    def request(self,method,path,**kwargs):
        calls.append(path)
        if path=='/internal/applications':return {'applications':[{'id':'document-ocr','name':'扫描应用'}]}
        if path=='/public/search':return {'results':[{'document_title':'旧版依据','version':'v2.9.1','content':'old version evidence'}]}
        raise AssertionError(path)
    monkeypatch.setattr('services.internal_client.InternalClient._request',request)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py').run()
    app.text_area[0].set_value('OCR 结果是什么？').run()
    next(b for b in app.button if b.label=='检索原文').click().run()
    assert any('old version evidence' in x.value for x in app.code)
    next(s for s in app.selectbox if s.label=='组件资料版本').set_value('v3.0.0').run()
    assert not app.exception
    assert not any('old version evidence' in x.value for x in app.code)
    assert calls.count('/public/search')==1


def test_configuration_submission_requires_pair_and_does_not_enable_model_advice(monkeypatch):
    monkeypatch.setenv('APP_ENV','internal')
    monkeypatch.setattr('internal_workbench.identity_header',lambda:'Bearer test-only')
    submitted=[]
    def request(self,method,path,**kwargs):
        if path=='/internal/applications':return {'applications':[{'id':'document-ocr','name':'扫描应用'}]}
        if method=='POST' and path=='/internal/tasks':
            submitted.append(kwargs['json']);return {'id':'a'*32}
        return {'state':'QUEUED','stage':'queued','attempts':0}
    monkeypatch.setattr('services.internal_client.InternalClient._request',request)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py').run()
    app.radio[0].set_value('变更审查').run()
    next(s for s in app.selectbox if s.label=='审查目的').set_value('同版本配置变更').run()
    assert not app.exception
    assert any('v2.9.1' in x.value and 'drop_score' in x.value for x in app.caption)
    button=next(b for b in app.button if b.label=='提交配置审查')
    button.click().run()
    assert not submitted and app.warning
    next(x for x in app.text_input if x.label=='基准应用提交（配置审查必填）').set_value('before').run()
    next(b for b in app.button if b.label=='提交配置审查').click().run()
    assert not app.exception
    assert submitted==[{'application_id':'document-ocr','head':'HEAD','base':'before',
                       'workflow_type':'configuration_change','generate_advice':False}]


def test_completed_configuration_task_uses_configuration_report(monkeypatch):
    import components.configuration_workbench as configuration_ui
    monkeypatch.setenv('APP_ENV','internal')
    monkeypatch.setattr('internal_workbench.identity_header',lambda:'Bearer test-only')
    rendered=[]
    monkeypatch.setattr(configuration_ui,'render_configuration_findings',lambda value,**kwargs:rendered.append(value),raising=False)
    def request(self,method,path,**kwargs):
        if path=='/internal/applications':return {'applications':[{'id':'document-ocr','name':'扫描应用'}]}
        if path.endswith('/report'):return {'report':{'workflow_type':'configuration_change','report_fingerprint':'b'*64},
            'records':[], 'regression_coverage':[{'label':'空页输出','status':'NOT_RUN'}]}
        return {'state':'DONE','stage':'completed','attempts':1,'error':None}
    monkeypatch.setattr('services.internal_client.InternalClient._request',request)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py').run()
    app.radio[0].set_value('变更审查').run()
    next(x for x in app.text_input if x.label=='任务编号').set_value('a'*32).run()
    assert not app.exception
    assert rendered and rendered[-1]['workflow_type']=='configuration_change'
    assert any(item.value == '逐项回归结果' for item in app.subheader)


def test_internal_joint_search_uses_commit_reference_and_hides_stale_result(monkeypatch):
    monkeypatch.setenv('APP_ENV', 'internal')
    monkeypatch.setattr('internal_workbench.identity_header', lambda: 'Bearer test-only')
    submitted = []

    def request(self, method, path, **kwargs):
        if path == '/internal/applications':
            return {'applications': [{'id': 'document-ocr', 'name': '扫描应用',
                                      'entrypoints': ['application/pipeline.py::process']}]}
        assert path == '/internal/application-knowledge'
        submitted.append(kwargs['json'])
        return {'status': 'EVIDENCE_ONLY', 'application_commit': 'a'*40, 'evidence': [
            {'document_title': '应用契约', 'content': 'page_index'}]}

    monkeypatch.setattr('services.internal_client.InternalClient._request', request)
    page = AppTest.from_file(Path(__file__).parents[1] / 'app.py').run()
    next(x for x in page.checkbox if x.label == '关联应用提交').check().run()
    next(x for x in page.selectbox if x.label == '应用入口').set_value('application/pipeline.py::process').run()
    page.text_area[0].set_value('OCR 输出契约').run()
    next(x for x in page.button if x.label == '检索原文').click().run()
    assert submitted == [{'application_id': 'document-ocr', 'head': 'HEAD', 'version': 'auto',
                          'entrypoint': 'application/pipeline.py::process',
                          'query': 'OCR 输出契约', 'generate': False}]
    assert any(x.value == 'page_index' for x in page.code)
    next(x for x in page.text_input if x.label == '应用提交 / 分支').set_value('candidate').run()
    assert not page.exception
    assert not any(x.value == 'page_index' for x in page.code)
