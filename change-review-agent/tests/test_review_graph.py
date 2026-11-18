from test_paddleocr_review import FILES, Gateway
from app.paddleocr_review import PaddleOCRReviewAgent


def test_compatibility_executes_real_graph_and_routes_optional_advice():
    for advice in (False, True):
        result = PaddleOCRReviewAgent(Gateway()).analyze(FILES, generate_advice=advice)
        execution = result['graph_execution']
        assert execution['engine'] == 'langgraph'
        assert execution['completed_nodes'][:4] == ['validate_input', 'static_review', 'evidence_validation', 'official_lookup']
        assert ('model_advice' in execution['completed_nodes']) is advice
        assert execution['completed_nodes'][-1] == 'report'
        assert execution['status'] == 'completed'


def test_checkpoint_state_excludes_runtime_objects_and_preserves_node_boundaries():
    import json
    from langgraph.checkpoint.memory import InMemorySaver

    class Runtime:
        def run(self, name, version, builder, initial):
            self.graph = builder(checkpointer=InMemorySaver())
            self.config = {'configurable': {'thread_id': 'review-1'}}
            result = self.graph.invoke(initial, self.config)
            state = self.graph.get_state(self.config)
            json.dumps(state.values, ensure_ascii=False)
            assert not state.next
            assert set(self.graph.get_graph().nodes) >= {'static_review', 'evidence_validation', 'official_lookup'}
            return result

    runtime = Runtime()
    result = PaddleOCRReviewAgent(Gateway()).analyze(FILES, graph_runtime=runtime)
    assert result['graph_execution']['completed_nodes'][-1] == 'report'


def test_configuration_conditional_skips_lookup_without_supported_changes():
    import hashlib
    from app.paddleocr_configuration_review import review_configuration_change

    class ConfigurationGateway(Gateway):
        def configuration_review(self, before_files, after_files, *, dependency_version):
            inventory = lambda files: [{'path': f['path'], 'sha256': hashlib.sha256(f['content'].encode()).hexdigest()} for f in files]
            return {'workflow_type': 'configuration_change', 'runtime_verified': False,
                    'dependency_version': dependency_version, 'before_files': inventory(before_files),
                    'files': inventory(after_files), 'changes': [], 'gaps': []}
        def search(self, *args, **kwargs):
            raise AssertionError('empty delta must not retrieve')

    result = review_configuration_change(ConfigurationGateway(), FILES, FILES)
    assert result['graph_execution']['completed_nodes'] == ['validate_input', 'static_review', 'evidence_validation', 'report']
    assert result['status'] == 'needs_verification'


def test_invalid_evidence_stops_graph_before_lookup_or_advice():
    import pytest
    gateway = Gateway()
    gateway.payload['findings'][0]['evidence'][0]['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='哈希'):
        PaddleOCRReviewAgent(gateway).analyze(FILES, generate_advice=True)
    assert not any(call[0] in ('search', 'model') for call in gateway.calls)


def test_real_review_graph_replays_durable_json_checkpoint_without_tool_calls(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'versioned-rag-service'))
    from src.graph_runtime import TaskGraphRuntime
    from src.internal_tasks import TaskStore
    store = TaskStore('sqlite:///' + str(tmp_path / 'tasks.db'))
    store.submit('alice', 'document-ocr', 'a' * 64, {'files': FILES})
    claim = store.claim('worker')
    gateway = Gateway()
    runtime = TaskGraphRuntime(store, claim, cancelled=lambda: False)
    first = PaddleOCRReviewAgent(gateway).analyze(FILES, graph_runtime=runtime)
    gateway.calls.clear()
    second = PaddleOCRReviewAgent(gateway).analyze(FILES, graph_runtime=runtime)
    assert first == second
    assert not gateway.calls


def test_inherited_langsmith_environment_cannot_export_review(monkeypatch):
    from unittest.mock import Mock
    from langsmith import Client
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'true')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'test-key-must-never-be-used')
    create = Mock(side_effect=AssertionError('unauthorized trace export'))
    update = Mock(side_effect=AssertionError('unauthorized trace export'))
    monkeypatch.setattr(Client, 'create_run', create)
    monkeypatch.setattr(Client, 'update_run', update)
    result = PaddleOCRReviewAgent(Gateway()).analyze(FILES)
    assert result['graph_execution']['status'] == 'completed'
    create.assert_not_called()
    update.assert_not_called()


def test_real_review_resumes_evidence_node_without_repeating_static_tool(tmp_path):
    import sys
    from pathlib import Path
    import pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'versioned-rag-service'))
    from src.graph_runtime import TaskGraphRuntime
    from src.internal_tasks import TaskStore
    store = TaskStore('sqlite:///' + str(tmp_path / 'tasks.db'))
    store.submit('alice', 'document-ocr', 'a' * 64, {'files': FILES})
    claim = store.claim('worker')

    class Interrupted(Gateway):
        def document(self, document_id):
            raise RuntimeError('worker stopped during read-only source validation')

    first = Interrupted()
    with pytest.raises(RuntimeError, match='worker stopped'):
        PaddleOCRReviewAgent(first).analyze(FILES, graph_runtime=TaskGraphRuntime(store, claim, cancelled=lambda: False))
    assert sum(call[0] == 'tool' for call in first.calls) == 1
    second = Gateway()
    result = PaddleOCRReviewAgent(second).analyze(FILES, graph_runtime=TaskGraphRuntime(store, claim, cancelled=lambda: False))
    assert not any(call[0] == 'tool' for call in second.calls)
    assert result['graph_execution']['completed_nodes'] == ['validate_input', 'static_review', 'evidence_validation', 'official_lookup', 'report']
