def test_missing_wrapper_file_cannot_prove_result_flow():
    from src.paddleocr_application_graph import analyze_application
    graph=analyze_application([{'path':'app/main.py','content':'from .missing import recognize\nresult = recognize("page.png")\n'}])
    assert any(g['code']=='missing_local_module' for g in graph['gaps'])
    assert not any(e['kind']=='result_flow' for e in graph['edges'])


def test_proven_direct_argument_flow_is_separate_from_reference():
    from src.paddleocr_application_graph import analyze_application, trace_impacts
    files=[{'path':'app/ocr_client.py','content':'def recognize(path):\n    return [[path]]\n'},
           {'path':'app/consumer.py','content':'def normalize(raw):\n    return raw[0]\n'},
           {'path':'app/main.py','content':'from .ocr_client import recognize\nfrom .consumer import normalize\ndef process(path):\n    raw = recognize(path)\n    return normalize(raw)\n'}]
    graph=analyze_application(files)
    assert any(e['kind']=='result_flow' and e['source'].endswith('::recognize') and e['target'].endswith('::normalize') for e in graph['edges'])
    impacts=trace_impacts(graph,[{'finding_id':'f1','status':'supported_risk','application':{'path':'app/ocr_client.py','line_start':2,'line_end':2}}])
    assert impacts and all(i['level']=='candidate' for i in impacts)
    assert any(i['flow_proven'] for i in impacts)


def test_shadowed_import_and_conditional_flow_not_proven():
    from src.paddleocr_application_graph import analyze_application
    files=[{'path':'app/w.py','content':'def recognize(x):\n    return x\n'},
           {'path':'app/main.py','content':'from .w import recognize\ndef process(recognize):\n    return recognize("x")\n'}]
    graph=analyze_application(files)
    assert not any(e['kind']=='result_flow' for e in graph['edges'])
    assert any(g['code']=='shadowed_symbol' for g in graph['gaps'])


def test_cyclic_calls_stop_and_record_coverage_gap():
    from src.paddleocr_application_graph import analyze_application, trace_impacts
    graph=analyze_application([{'path':'app.py','content':'def a():\n    return b()\ndef b():\n    return a()\n'}])
    paths=trace_impacts(graph,[{'finding_id':'f','application':{'path':'app.py','line_start':1,'line_end':2}}])
    assert len(paths)<10
    assert any(g['code']=='cyclic_application_relation' for g in graph['gaps'])


def test_module_rebinding_invalidates_deferred_function_lookup():
    from src.paddleocr_application_graph import analyze_application
    graph=analyze_application([{'path':'w.py','content':'def recognize(x):\n    return x\n'},
        {'path':'main.py','content':'from w import recognize\nrecognize = unknown_wrapper\ndef process(x):\n    return recognize(x)\n'}])
    assert not any(e['kind']=='result_flow' for e in graph['edges'])


def test_real_review_traces_basic_entry_to_downstream_without_claiming_breakage():
    from pathlib import Path
    from src.public_knowledge import PublicKnowledgeIndex
    from src.paddleocr_compatibility import review_compatibility
    root=Path(__file__).resolve().parents[1]/'public_corpus_paddleocr'
    files=[{'path':'app/ocr_client.py','content':'from paddleocr import PaddleOCR\ndef recognize(path):\n    engine = PaddleOCR(lang="ch")\n    return engine.ocr(path)\n'},
        {'path':'app/main.py','content':'from .ocr_client import recognize\ndef process(path):\n    raw = recognize(path)\n    return raw\n'}]
    report=review_compatibility(PublicKnowledgeIndex(root),source_version='v2.9.1',target_version='v3.0.0',files=files)
    assert report['impact_paths']
    assert all(p['level']=='candidate' and not p['runtime_verified'] for p in report['impact_paths'])
