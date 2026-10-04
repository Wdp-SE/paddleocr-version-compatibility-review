"""Publish only current-build source-derived acceptance, never a quality score."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_paddleocr_acceptance(corpus: Path) -> dict | None:
    evaluation = ROOT / "evaluation" / "paddleocr_compatibility_v1"
    try:
        report = _read(evaluation / "report.json")
        cases = _read(evaluation / "cases.json")
        manifest = _read(corpus / "corpus_manifest.json")
        bindings = {
            "cases_sha256": evaluation / "cases.json", "runner_sha256": evaluation / "run_evaluation.py",
            "corpus_manifest_sha256": corpus / "corpus_manifest.json", "chunks_sha256": corpus / "chunks.json",
            "retrieval_code_sha256": ROOT / "versioned-rag-service/src/public_knowledge.py",
            "fusion_code_sha256": ROOT / "versioned-rag-service/src/retrieval_fusion.py",
            "compatibility_code_sha256": ROOT / "versioned-rag-service/src/paddleocr_compatibility.py",
        }
        if (manifest.get("workspace_id") != "paddleocr" or report.get("project_id") != "paddleocr"
                or report.get("assessment_type") != "source_derived_acceptance"
                or report.get("dataset_id") != cases["dataset_id"]
                or report.get("top_k") != cases["top_k"]
                or report.get("answer_accuracy") is not None
                or report.get("independent_business_accuracy") is not None
                or any(report.get(key) != _sha(path) for key, path in bindings.items())):
            return None
        expected_ids = [case["id"] for case in cases["retrieval_cases"]]
        for values in report["retrieval"].values():
            measured = values["cases"]
            if (values["count"] != len(expected_ids) or [row["id"] for row in measured] != expected_ids
                    or any(type(row["matched"]) is not bool for row in measured)
                    or values["evidence_span_hit_count"] != sum(row["matched"] for row in measured)
                    or any(type(row["wrong_version_count"]) is not int or row["wrong_version_count"] < 0 for row in measured)
                    or values["wrong_version_count"] != sum(row["wrong_version_count"] for row in measured)):
                return None
        measured = report["compatibility"]
        expected = cases["compatibility_cases"]
        if len(measured) != len(expected):
            return None
        for row, case in zip(measured, expected):
            sample = ROOT / "examples/paddleocr_document_app" / case["file"]
            if (row["id"] != case["id"] or row["application_sha256"] != _sha(sample)
                    or row["runtime_verified"] is not False or type(row["passed"]) is not bool):
                return None
            actual_pass = row["status"] == case["expected_status"] and case["expected_rule"] in row["rules"]
            if row["passed"] != actual_pass:
                return None
        return report
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def load_quality_comparison(corpus: Path) -> dict | None:
    """Bind comparison to its corpus, metric implementation and retrieval build."""
    from .quality_evaluation import assess_retrieval, select_strategy
    folder=ROOT/'evaluation/paddleocr_quality_v2'
    try:
        report=_read(folder/'report.json')
        cases=_read(folder/'cases.json')
        bindings={'cases_sha256':folder/'cases.json','chunks_sha256':corpus/'chunks.json',
                  'corpus_manifest_sha256':corpus/'corpus_manifest.json',
                  'retriever_sha256':ROOT/'versioned-rag-service/src/paddleocr_quality.py',
                  'metric_code_sha256':ROOT/'versioned-rag-service/src/quality_evaluation.py',
                  'lexical_code_sha256':ROOT/'versioned-rag-service/src/public_knowledge.py',
                  'fusion_code_sha256':ROOT/'versioned-rag-service/src/retrieval_fusion.py',
                  'runner_sha256':folder/'run_evaluation.py'}
        if (report['dataset_id']!=cases['dataset_id'] or report['top_k']!=cases['top_k']
                or report['answer_accuracy'] is not None or report['business_impact_accuracy'] is not None
                or any(report.get(key)!=_sha(path) for key,path in bindings.items())):
            return None
        chunks={c['chunk_id']:c for c in _read(corpus/'chunks.json')}
        expected=sorted(cases['cases'],key=lambda c:c['split']!='dev')
        for policy,values in report['strategies'].items():
            rows=values['cases']
            if len(rows)!=len(expected):return None
            for row,case in zip(rows,expected):
                ids=row['chunk_ids']
                if (row['id']!=case['id'] or row['split']!=case['split']
                        or len(ids)>report['top_k'] or len(set(ids))!=len(ids)):
                    return None
                metrics=assess_retrieval(case,[chunks[cid] for cid in ids])
                if any(row[key]!=value for key,value in metrics.items()):return None
            for split in ('dev','holdout'):
                subset=[row for row in rows if row['split']==split]
                summary=values[split]
                if (summary['count']!=len(subset)
                        or summary['complete_rate']!=sum(row['complete'] for row in subset)/len(subset)
                        or summary['fact_recall']!=sum(row['matched_fact_count'] for row in subset)/sum(row['required_fact_count'] for row in subset)
                        or summary['wrong_version_count']!=sum(row['wrong_version_count'] for row in subset)):
                    return None
        if report['selected_on_dev']!=select_strategy(report['strategies']):return None
        return report
    except (OSError,ValueError,KeyError,TypeError,AttributeError,ZeroDivisionError):
        return None


def load_upgrade_demo() -> dict | None:
    """This original application was executed; arbitrary review inputs were not."""
    folder=ROOT/'evaluation/paddleocr_impact_v3'
    application=ROOT/'examples/paddleocr_document_app'
    try:
        reports=[_read(folder/f'runtime-v{i}.json') for i in (2,3)]
        input_hashes=[_sha(application/'fixtures'/f'page-{i}.png') for i in (1,2,3)]
        for report,expected_version in zip(reports,('2.9.1','3.0.0')):
            if (report['paddleocr_version']!=expected_version
                    or report['application_sha256']!=_sha(application/'run_upgrade_demo.py')
                    or report['adapter_sha256']!=_sha(application/'result_adapter.py')
                    or report['consumer_sha256']!=_sha(application/'application/consumer.py')
                    or report['downstream_contract_passed'] is not True
                    or report['input_sha256']!=input_hashes
                    or report['sample_regression_passed'] is not True
                    or len(report['normalization_checks'])!=3
                    or any(row['passed'] is not True for row in report['normalization_checks'])):
                return None
            actual={str(p.relative_to(application)).replace('\\','/'): _sha(p)
                    for p in (application/'application').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            if report['application_files_sha256']!=actual:return None
            from importlib.util import spec_from_file_location,module_from_spec
            spec=spec_from_file_location('verified_original_consumer',application/'application/consumer.py')
            module=module_from_spec(spec);spec.loader.exec_module(module)
            if module.to_json(report['normalized_results'])!=report['downstream_json']:return None
        if reports[0]['legacy_consumer_success'] is not True or reports[1]['legacy_consumer_success'] is not False:
            return None
        return {'assessment_type':'original_application_runtime_regression',
                'uploaded_application_runtime_verified':False,'reports':reports,
                'scope':'原创样例三页回归；不是 OCR 准确率，也不证明用户应用兼容。'}
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None


def load_independent_probes(corpus: Path) -> dict | None:
    from .quality_evaluation import assess_retrieval
    folder=ROOT/'evaluation/paddleocr_quality_v2'
    try:
        report=_read(folder/'independent-report.json')
        probes=_read(folder/'independent-probes.json')
        comparison=load_quality_comparison(corpus)
        bindings={'comparison_sha256':folder/'report.json','probes_sha256':folder/'independent-probes.json',
                  'chunks_sha256':corpus/'chunks.json','runner_sha256':folder/'run_independent_probes.py',
                  'static_code_sha256':ROOT/'versioned-rag-service/src/paddleocr_compatibility.py'}
        if (comparison is None or any(report.get(k)!=_sha(p) for k,p in bindings.items())
                or report['selected_strategy_frozen_before_probes']!=comparison['selected_on_dev']
                or len(report['retrieval'])!=len(probes['retrieval_cases'])
                or len(report['static'])!=len(probes['static_upgrade_cases'])):
            return None
        chunks={c['chunk_id']:c for c in _read(corpus/'chunks.json')}
        for row,case in zip(report['retrieval'],probes['retrieval_cases']):
            metrics=assess_retrieval(case,[chunks[cid] for cid in row['chunk_ids']])
            if row['id']!=case['id'] or any(row[k]!=v for k,v in metrics.items()):return None
        for row,case in zip(report['static'],probes['static_upgrade_cases']):
            passed=(row['status']==case['expected_status'] and row['runtime_verified'] is False
                    and set(case['required_rules'])<=set(row['rules'])
                    and set(case['required_gap_codes'])<=set(row['gaps'])
                    and not set(case.get('forbidden_rules',[]))&set(row['rules']))
            if row['id']!=case['id'] or row['passed']!=passed:return None
        return {'assessment_type':report['assessment_type'],
                'retrieval_complete':sum(r['complete'] for r in report['retrieval']),
                'retrieval_count':len(report['retrieval']),
                'static_passed':sum(r['passed'] for r in report['static']),
                'static_count':len(report['static']),'limitations':report['limitations']}
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None
