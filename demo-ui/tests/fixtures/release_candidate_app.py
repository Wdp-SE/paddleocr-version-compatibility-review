import streamlit as st
import public_workbench as workbench

if st.session_state.get("test_panel") == "configuration":
    workbench._configuration_trace_panel({
        "configuration_trace": {"status": "UNAVAILABLE", "relations": [], "gaps": []},
    })
elif st.session_state.get("test_panel") == "unresolved":
    workbench._configuration_trace_panel({
        "configuration_trace": {"status": "partial", "relations": [],
            "gaps": [{"kind": "unsupported_base_syntax"}], "bounds": {"truncated": True}},
    })
elif st.session_state.get("test_panel") == "answer_gaps":
    workbench._answer_gaps_panel({"evidence_gaps": ["未提供目标模型的精度和跟踪回归记录。"]})
else:
    workbench._pphuman_development_evaluation({
        "development_evaluation": {
            "assessment_type": "dev_diagnostic_only",
            "default_promotion_eligible": False,
            "denominators": {"rag_answerable": 11, "rag_unanswerable": 1, "agent_answerable": 2},
            "rag": {"bm25@5": {"top_k": 5, "mean_required_source_recall": 0.5}},
            "agent": {"bm25": {"required_evidence_source_recall": 0.5}},
        },
    })
