import json
from pathlib import Path

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
    path = Path(__file__).resolve().parents[3] / "evaluation/pphuman_release_candidate_v1/report.json"
    workbench._pphuman_development_evaluation({
        "development_evaluation": json.loads(path.read_text(encoding="utf-8")),
    })
