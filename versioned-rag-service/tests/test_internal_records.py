import copy
import time
import pytest
from src.internal_tasks import fingerprint
from src.internal_records import validate_ci_record,validate_decision


@pytest.fixture
def record():
    task={'state':'DONE','owner':'alice','request':{'snapshot':{'head_commit':'a'*40,
        'inventory':[{'path':'application/contract.json','sha256':'b'*64},{'path':'application/tests/test_contract.py','sha256':'c'*64}]},
        'source_version':'v2.9.1','target_version':'v3.0.0'},'result':{'runtime_verified':False,'report_fingerprint':'d'*64}}
    claims={'repository':'company/ocr','sha':'a'*40,'job_workflow_ref':'company/ocr/.github/workflows/trusted.yml@refs/heads/main',
        'job_workflow_sha':'e'*40,'run_id':'123','run_attempt':'1','run_number':'12',
        'workflow_ref':'company/ocr/.github/workflows/checks.yml@refs/heads/main'}
    policy={'ci':{'repository':claims['repository'],'workflow_ref':claims['job_workflow_ref'],'workflow_sha':'e'*40}}
    payload={'application_commit':'a'*40,'report_fingerprint':'d'*64,'run_id':'123','status':'PASSED',
        'input_sha256':{'application/contract.json':'b'*64,'application/tests/test_contract.py':'c'*64},
        'environment':{'paddleocr':'3.0.0','paddlepaddle':'3.0.0','python':'3.11','runner_os':'linux',
                       'model_sha256':'f'*64,'container_digest':'sha256:'+'1'*64}}
    return task,payload,claims,policy


def test_ci_record_binds_actual_identity_workflow_inputs_and_environment(record):
    task,payload,claims,policy=record
    checked=validate_ci_record(task,payload,claims,policy)
    assert checked['provenance']=='trusted_ci_identity' and checked['upgrade_approved'] is False


@pytest.mark.parametrize('field,value',[('sha','0'*40),('job_workflow_sha','0'*40),('repository','foreign/repo')])
def test_ci_claim_changes_are_rejected(record,field,value):
    task,payload,claims,policy=record;claims[field]=value
    with pytest.raises(ValueError):validate_ci_record(task,payload,claims,policy)


def test_old_report_or_missing_test_hash_is_rejected(record):
    task,payload,claims,policy=record;payload['input_sha256'].pop('application/tests/test_contract.py')
    with pytest.raises(ValueError):validate_ci_record(task,payload,claims,policy)


def test_decision_requires_reviewer_and_binds_current_report(record):
    task,*_=record
    payload={'decision':'ACKNOWLEDGED','report_fingerprint':'d'*64,'reason':'已检查静态报告并安排开发者完成回归验证'}
    checked=validate_decision(task,payload,'reviewer',[])
    assert checked['actor']=='reviewer' and checked['upgrade_approved'] is False
    payload['report_fingerprint']='0'*64
    with pytest.raises(ValueError):validate_decision(task,payload,'reviewer',[])


def test_continue_requires_independent_reviewer_and_target_ci(record):
    task,payload,claims,policy=record
    decision={'decision':'CONTINUE_REVIEW','report_fingerprint':'d'*64,'reason':'回归记录已核对并继续人工升级流程',
              'expires_at':time.time()+3600,'accepted_risks':[]}
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',[])
    ci=validate_ci_record(task,payload,claims,policy)
    with pytest.raises(ValueError):validate_decision(task,decision,'alice',[{'kind':'CI','payload':ci}])
    assert not validate_decision(task,decision,'reviewer',[{'kind':'CI','payload':ci}])['upgrade_approved']


def test_later_failed_target_ci_invalidates_prior_pass(record):
    task,payload,claims,policy=record
    passed=validate_ci_record(task,payload,claims,policy)
    claims['run_id']='124';claims['run_number']='13';payload['run_id']='124';payload['status']='FAILED'
    failed=validate_ci_record(task,payload,claims,policy)
    decision={'decision':'CONTINUE_REVIEW','report_fingerprint':'d'*64,'reason':'核对当前运行记录后继续人工升级评审',
        'expires_at':time.time()+3600,'accepted_risks':[]}
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',
        [{'kind':'CI','payload':passed},{'kind':'CI','payload':failed}])
    # An older success arriving late must not override the newer failed run.
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',
        [{'kind':'CI','payload':failed},{'kind':'CI','payload':passed}])


def test_rerun_attempt_failure_invalidates_same_run_prior_pass(record):
    task,payload,claims,policy=record
    passed=validate_ci_record(task,payload,claims,policy)
    claims['run_attempt']='2';payload['status']='FAILED'
    failed=validate_ci_record(task,payload,claims,policy)
    decision={'decision':'CONTINUE_REVIEW','report_fingerprint':'d'*64,'reason':'核对当前运行记录后继续人工升级评审',
        'expires_at':time.time()+3600,'accepted_risks':[]}
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',
        [{'kind':'CI','payload':failed},{'kind':'CI','payload':passed}])


def test_ci_order_uses_workflow_sequence_not_opaque_run_id(record):
    task,payload,claims,policy=record
    passed=validate_ci_record(task,payload,claims,policy)
    claims.update(run_id='122',run_number='13');payload.update(run_id='122',status='FAILED')
    failed=validate_ci_record(task,payload,claims,policy)
    decision={'decision':'CONTINUE_REVIEW','report_fingerprint':'d'*64,'reason':'核对当前运行记录后继续人工升级评审',
        'expires_at':time.time()+3600,'accepted_risks':[]}
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',
        [{'kind':'CI','payload':failed},{'kind':'CI','payload':passed}])


def test_malformed_finding_reference_is_validation_error(record):
    task,*_=record
    task['result']['compatibility_report']={'findings':[{'finding_id':'risk-1','status':'supported_risk'}]}
    decision={'decision':'ACKNOWLEDGED','report_fingerprint':'d'*64,'reason':'记录本次静态审查的人工核查意见',
        'accepted_risks':[{'finding_id':{},'reason':'无效引用不能写入审查记录'}]}
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',[])


def test_configuration_continue_uses_same_version_ci_and_records_behavior_acceptance(record):
    task,payload,claims,policy=record
    task['request'].update(workflow_type='configuration_change',target_version='v2.9.1')
    task['result']['compatibility_report']={'findings':[{'finding_id':'threshold-1','status':'behavior_change'}]}
    payload['environment']['paddleocr']='2.9.1'
    ci=validate_ci_record(task,payload,claims,policy)
    decision={'decision':'CONTINUE_REVIEW','report_fingerprint':'d'*64,
        'reason':'已核对阈值变更及同版本回归，继续人工评审','expires_at':time.time()+3600,
        'accepted_risks':[{'finding_id':'threshold-1','reason':'已确认过滤阈值提高符合本应用业务要求'}]}
    checked=validate_decision(task,decision,'reviewer',[{'kind':'CI','payload':ci}])
    assert checked['upgrade_approved'] is False
    wrong=copy.deepcopy(ci);wrong['environment']['paddleocr']='3.0.0'
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',[{'kind':'CI','payload':wrong}])
    decision['accepted_risks']=[]
    with pytest.raises(ValueError):validate_decision(task,decision,'reviewer',[{'kind':'CI','payload':ci}])


def test_regression_coverage_uses_latest_attempt_and_preserves_missing_checks(record):
    from src.internal_records import regression_coverage
    task, payload, claims, policy = record
    task['result']['compatibility_report'] = {'regression_requirements': [
        {'finding_id': 'output-contract', 'checks': ['空页保留', 'JSON 消费']}]}
    rows = regression_coverage(task, [])
    assert [row['status'] for row in rows] == ['NOT_RUN', 'NOT_RUN']
    payload['check_results'] = [{'check_id': rows[0]['check_id'], 'status': 'PASSED',
                                 'execution_kind': 'REAL_OCR', 'artifact_sha256': 'a'*64}]
    early = validate_ci_record(task, payload, claims, policy)
    assert [row['status'] for row in regression_coverage(task, [{'kind': 'CI', 'payload': early}])] == ['PASSED', 'NOT_RUN']
    payload['check_results'] = [{**payload['check_results'][0], 'check_id': rows[1]['check_id'],
                                 'execution_kind': 'FIXTURE'}]
    later = validate_ci_record(task, payload, {**claims, 'run_attempt': '2'}, policy)
    records = [{'kind': 'CI', 'payload': ci} for ci in (later, early)]
    assert [row['status'] for row in regression_coverage(task, records)] == ['NOT_RUN', 'FIXTURE_PASSED']
    with pytest.raises(ValueError):
        validate_ci_record(task, {**payload, 'check_results': [{**payload['check_results'][0],
                           'check_id': 'unknown'}]}, claims, policy)
