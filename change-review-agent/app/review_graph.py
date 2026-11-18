"""Typed LangGraph orchestration. Runtime objects live only in node closures."""
from typing import Callable, TypedDict
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

GRAPH_VERSION = 1

class ReviewData(TypedDict, total=False):
    files: list[dict]
    before_files: list[dict]
    after_files: list[dict]
    source_version: str
    target_version: str
    dependency_version: str
    generate_advice: bool
    resume: dict | None
    submitted: dict[str, str]
    before: dict[str, str]
    after: dict[str, str]
    workspace: dict
    task_identity: str
    report: dict
    tool: dict
    elapsed_seconds: float
    stage_latency: dict[str, float]
    file_rows: list[dict]
    findings: list[dict]
    registry: dict[str, dict]
    chunks_by_source: dict[str, list[dict]]
    checked: dict[str, dict]
    gaps: list[dict]
    expected_summary: dict[str, int]
    investigation: dict
    related: dict[str, dict]
    application_validation: list[dict]
    lookup_gaps: list[str]
    model_calls: int
    model_suggestions: list[dict]
    model_status: str
    model_generation: dict
    support_calls: int | None
    advice_cache: dict | None
    target_evidence: dict[str, dict]
    evidence: list[dict]
    dependency_verified: bool
    result: dict


class ReviewState(TypedDict):
    data: ReviewData
    completed_nodes: list[str]


def build_review_graph(nodes: dict[str, Callable], *, advice_route=None, lookup_route=None, checkpointer=None):
    graph = StateGraph(ReviewState)
    for name, action in nodes.items():
        graph.add_node(name, action)
    graph.add_edge(START, 'validate_input')
    graph.add_edge('validate_input', 'static_review')
    graph.add_edge('static_review', 'evidence_validation')
    if lookup_route:
        graph.add_conditional_edges('evidence_validation',
            lambda state: 'official_lookup' if lookup_route(state) else 'report',
            {'official_lookup': 'official_lookup', 'report': 'report'})
    else:
        graph.add_edge('evidence_validation', 'official_lookup')
    if 'model_advice' in nodes:
        graph.add_conditional_edges('official_lookup',
            lambda state: 'model_advice' if advice_route(state) else 'report',
            {'model_advice': 'model_advice', 'report': 'report'})
        graph.add_edge('model_advice', 'report')
    else:
        graph.add_edge('official_lookup', 'report')
    graph.add_edge('report', END)
    return graph.compile(checkpointer=checkpointer)


def run_review_graph(workflow, nodes, initial_state, *, advice_route=None, lookup_route=None, runtime=None):
    def build_graph(checkpointer=None):
        return build_review_graph(nodes, advice_route=advice_route, lookup_route=lookup_route, checkpointer=checkpointer)
    # Application text may only leave through the explicitly configured gateway.
    # Ignore inherited LangSmith/LangChain tracing environment switches.
    with tracing_context(enabled=False):
        final = (build_graph().invoke(initial_state) if runtime is None else
                 runtime.run(workflow, GRAPH_VERSION, build_graph, initial_state))
    result = final['data']['result']
    result['graph_execution'] = {'engine': 'langgraph', 'workflow': workflow, 'graph_version': GRAPH_VERSION,
                               'nodes': list(nodes), 'completed_nodes': final['completed_nodes'], 'status': 'completed'}
    return result
