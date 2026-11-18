import copy
from pathlib import Path
from streamlit.testing.v1 import AppTest
from services.public_knowledge_client import PublicKnowledgeClient


def test_knowledge_page_loads_registered_commit_and_submits_both_sources(monkeypatch, tmp_path):
    from services import registered_application as registered
    monkeypatch.setenv('APP_ENV','public_demo')
    monkeypatch.setenv('DEMO_AUDIT_DB_PATH',str(tmp_path/'audit.db'))
    from test_paddleocr_workbench import WORKSPACE
    workspace=copy.deepcopy(WORKSPACE)
    monkeypatch.setattr(PublicKnowledgeClient,'workspace',lambda self:copy.deepcopy(workspace))
    monkeypatch.setattr(PublicKnowledgeClient,'documents',lambda self:[])
    monkeypatch.setattr(registered,'load_registry',lambda:{'ocr':{'id':'ocr','name':'原创OCR应用','source_identity':'student_poc_application'}})
    snapshot={'application_id':'ocr','head_commit':'a'*40,
              'files':[{'path':'application/contract.json','content':'{"output_fields":["pages"]}'}], 'gaps':[]}
    monkeypatch.setattr(registered,'collect_snapshot',lambda *a,**kw:copy.deepcopy(snapshot))
    calls=[]
    def query(self, files, question, **kw):
        calls.append((files,question,kw))
        return {'status':'GENERATION_NOT_CONFIGURED','sources':[],'evidence':[{
            'chunk_id':'app-contract','document_key':'application/contract.json','document_id':'application/contract.json',
            'document_path':'application/contract.json','source_type':'submitted_application',
            'source_identity':'submitted_application','version':'application@'+'a'*40,'application_commit':'a'*40,
            'content':files[0]['content'],'line_start':1,'line_end':1,'source_url':None,'locale':'zh-CN'}]}
    monkeypatch.setattr(PublicKnowledgeClient,'application_query',query)
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py',default_timeout=30).run()
    app.button(key='nav_版本检索与问答').click().run()
    app.button(key='knowledge_load_application').click().run()
    app.button(key='knowledge_generate').click().run()
    assert not app.exception
    assert calls[0][2]['application_commit']=='a'*40
    assert calls[0][0]==snapshot['files']
    captions='\n'.join(str(row.value) for row in app.caption)
    assert '应用提交资料' in captions
    assert 'aaaaaaaaaaaa' in captions
