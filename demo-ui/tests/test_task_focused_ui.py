import copy
from pathlib import Path
from streamlit.testing.v1 import AppTest


def sample():
    def finding(i,line,*,status='supported_risk',path='application/normalizer.py',version='v3.0.0'):
        return {'finding_id':i,'rule_id':'legacy_ocr_result','status':status,'title':'旧结果索引需要调整',
                'explanation':'旧结构与目标结果访问方式不同','desired_check':'调整结果适配并执行回归',
                'application':{'path':path,'line':line,'end_line':line,'snippet':'line[1][0]'},
                'evidence':[{'version':version,'document_id':'api','line_start':1,'line_end':10,
                    'url':'https://github.com/PaddlePaddle/PaddleOCR/blob/example/paddleocr.py',
                    'path':'paddleocr.py','title':'OCR 接口','commit':'b'*40,'sha256':'c'*64}]}
    return {'request_mode':'paddleocr_compatibility','source_version':'v2.9.1','target_version':'v3.0.0',
            'report_fingerprint':'a'*64,'status':'supported_risk','compatibility_report':{
                'source_version':'v2.9.1','target_version':'v3.0.0','status':'supported_risk',
                'findings':[finding('risk-1',22),finding('risk-2',24),finding('ok',5,status='unaffected')],
                'gaps':[{'code':'missing_call','detail':'缺调用层','application':{'path':'application/call.py'}}],
                'regression_requirements':[]},'runtime_verified':False}


def test_grouped_risks_preserve_every_id_and_do_not_count_unaffected_as_risk():
    from services.review_action_summary import finding_groups,review_overview
    result=sample();original=copy.deepcopy(result)
    groups=finding_groups(result,statuses=('supported_risk',))
    assert len(groups)==1 and [f['finding_id'] for f in groups[0]['findings']]==['risk-1','risk-2']
    assert len(groups[0]['locations'])==2
    overview=review_overview(result)
    assert overview['risk_count']==1 and overview['risk_location_count']==2
    assert overview['gap_count']==1 and overview['runtime_label']=='未关联回归记录'
    assert result==original
    # Different source versions or files cannot silently become one issue.
    result['compatibility_report']['findings'][1]['evidence'][0]['version']='v2.9.1'
    assert len(finding_groups(result,statuses=('supported_risk',)))==2


def test_navigation_has_six_primary_destinations_without_task_detail_duplicates(monkeypatch):
    monkeypatch.setenv('RAG_API_BASE_URL','http://127.0.0.1:1')
    monkeypatch.setenv('DEMO_REQUEST_TIMEOUT_SECONDS','0.1')
    monkeypatch.setenv('APP_ENV','public_demo')
    app=AppTest.from_file(Path(__file__).parents[1]/'app.py',default_timeout=30).run()
    assert not app.exception
    assert [b.label for b in app.sidebar.button if str(b.key).startswith('nav_')]==[
        '总览','知识查询','变更审查','资料库','评测报告','帮助']


def test_report_has_one_summary_and_task_tabs_not_three_progress_sections():
    app=AppTest.from_string('''
from public_workbench import _ocr_report
from tests.test_task_focused_ui import sample
_ocr_report(sample())
''',default_timeout=20).run()
    assert not app.exception
    assert [t.label for t in app.tabs]==['审查结果','证据与代码','回归要求']
    text='\n'.join(str(x.value) for x in [*app.markdown,*app.caption,*app.info,*app.warning])
    assert '本次审查进度' not in text
    assert '风险问题 1 项' in text and '2 个位置' in text
    assert '未关联回归记录' in text
    assert '最多 12 项' not in text
    summary=next(str(item.value) for item in app.markdown if 'class="review-summary"' in str(item.value))
    assert '\n    <div>' not in summary  # Markdown must not render the HTML as an indented code block.


def test_task_page_breadcrumb_uses_current_navigation_names():
    app=AppTest.from_string('''
from public_workbench import _page_header
_page_header("系统说明","帮助",page_key="help")
''').run()
    assert not app.exception
    assert [b.label for b in app.button]==['←','首页']
    assert any('帮助' in str(item.value) for item in app.markdown)


def test_summary_only_associates_ci_records_for_current_commit_report_and_dependency():
    from services.review_action_summary import review_overview
    result=sample();result['application_snapshot']={'head_commit':'b'*40}
    record={'kind':'CI','payload':{'provenance':'trusted_ci_identity','report_fingerprint':'a'*64,
            'application_commit':'b'*40,'environment':{'paddleocr':'3.0.0'},'status':'PASSED'}}
    assert review_overview(result,records=[record])['runtime_label']=='已关联可信 CI 记录'
    for field,value in [('application_commit','c'*40),('report_fingerprint','d'*64),('provenance','manual')]:
        invalid=copy.deepcopy(record);invalid['payload'][field]=value
        assert review_overview(result,records=[invalid])['runtime_label']=='未关联回归记录'
    invalid=copy.deepcopy(record);invalid['payload']['environment']['paddleocr']='2.9.1'
    assert review_overview(result,records=[invalid])['runtime_label']=='未关联回归记录'


def test_grouped_disposition_keeps_individual_decisions(monkeypatch):
    from services.paddleocr_disposition import binding
    result=sample();prefix='ocr_disposition_'+binding(result)[:16]
    app=AppTest.from_string('''
from types import SimpleNamespace
from tests.test_task_focused_ui import sample
from components.paddleocr_disposition import render_disposition
render_disposition(sample(),SimpleNamespace(),"test-session")
''').run()
    assert not app.exception
    assert app.selectbox(key=prefix+'risk-1_decision').value=='pending'
    assert app.selectbox(key=prefix+'risk-2_decision').value=='pending'
    app.selectbox(key=prefix+'risk-1_decision').set_value('needs_change').run()
    assert app.selectbox(key=prefix+'risk-2_decision').value=='pending'
    assert app.button(key=prefix+'_continue').disabled
