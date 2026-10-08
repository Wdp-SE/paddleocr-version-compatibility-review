"""Visual tokens for the official public engineering workbench."""

from base64 import b64encode
from pathlib import Path


def _frame_data_uri(filename: str) -> str:
    artwork = Path(__file__).resolve().parents[1] / "assets" / filename
    return "data:image/svg+xml;base64," + b64encode(artwork.read_bytes()).decode("ascii")


PUBLIC_CSS = """
<style>
:root {
  --paper:#ffffff; --canvas:#f5f5f5; --ink:#171717; --body:#303030;
  --muted:#656565; --line:#d8d8d8; --line-strong:#777777;
  --quiet:#f7f7f7; --signal:#484848;
}
.stApp {background:var(--canvas);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;}
[data-testid="stHeader"] {background:var(--paper);}
a {color:var(--ink)!important;text-decoration-color:var(--line-strong)!important;text-underline-offset:.15em;}
a:hover {text-decoration-color:var(--ink)!important;}
#MainMenu,footer {visibility:hidden;}
[data-testid="stAppDeployButton"] {display:none;}
.block-container {background:var(--paper);max-width:none!important;width:100%!important;margin:0!important;box-sizing:border-box;
  padding:3rem 1.25rem 1.5rem;overflow:visible;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.home-snapshot) {
  margin-top:1.15rem!important;border-top:1px solid var(--line);padding-top:.8rem;
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.home-snapshot) [data-testid="stMarkdownContainer"] {
  margin:0!important;
}
.home-snapshot {color:var(--muted);font-size:1.18rem;line-height:1.55;}
[data-testid="stMain"] {min-width:0!important;flex:1 1 0%!important;background:var(--paper);}
[data-testid="stSidebar"][aria-expanded="true"] {width:clamp(260px,17vw,330px)!important;min-width:clamp(260px,17vw,330px)!important;
  max-width:330px!important;flex:0 0 clamp(260px,17vw,330px)!important;
  background:var(--quiet);border-right:1px solid var(--line);}
[data-testid="stSidebar"][aria-expanded="false"] {width:0!important;min-width:0!important;max-width:0!important;
  flex:0 0 0!important;overflow:hidden!important;border-right:0!important;}
[data-testid="stSidebar"] [data-testid="stButton"] button {
  min-height:2.55rem;padding:.5rem .85rem;text-align:left;justify-content:flex-start;
  border-radius:4px;border:1px solid transparent;background:transparent;color:var(--body);
  font-size:1.3rem;font-weight:560;box-shadow:none;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {background:#ededed;color:var(--ink);}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
  background:var(--ink);color:#fff;border-color:var(--ink);font-weight:680;
}
h1 {font-size:2.45rem!important;line-height:1.22!important;letter-spacing:-.035em!important;
  color:var(--ink)!important;margin-bottom:.35rem!important;font-weight:760!important;}
h2 {font-size:1.86rem!important;letter-spacing:-.025em!important;color:var(--ink)!important;}
h3 {font-size:1.5rem!important;color:var(--ink)!important;}
p,li {font-size:clamp(1.14rem,1.06rem + .2vw,1.3rem);line-height:1.65;color:var(--body);}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {font-size:clamp(1.14rem,1.06rem + .2vw,1.3rem)!important;
  line-height:1.65!important;color:var(--body);}
[data-testid="stCaptionContainer"] {color:var(--muted)!important;font-size:1.2rem;line-height:1.55;}
[data-testid="stWidgetLabel"] p {font-size:1.12rem!important;line-height:1.45!important;}
[data-testid="stVerticalBlockBorderWrapper"] {
  background:var(--paper);border:1px solid var(--line);border-radius:5px;box-shadow:none;
}
[data-testid="stExpander"] {background:var(--paper);border:1px solid var(--line);border-radius:4px;}
[data-testid="stExpander"] summary p {font-size:1.14rem!important;line-height:1.45!important;}
[data-testid="stAlert"] {
  background:var(--paper)!important;background-color:var(--paper)!important;background-image:none!important;
  border:1px solid var(--line)!important;
  border-left:3px solid var(--line-strong)!important;border-radius:5px;box-shadow:none;
}
[data-testid="stAlert"] [data-baseweb="notification"],
[data-testid^="stAlertContent"],
[role="alert"] {
  background:var(--paper)!important;background-color:var(--paper)!important;background-image:none!important;
}
[data-testid="stAlert"] *,[role="alert"] * {
  background-color:transparent!important;background-image:none!important;
}
[data-testid="stAlert"] [data-baseweb="notification"],
[data-testid="stAlert"] [data-testid="stAlertContent"] {
  background:var(--paper)!important;color:var(--body)!important;border-radius:4px;
}
[data-testid="stAlert"] svg {color:var(--muted)!important;}
/* Preserve severity in a quiet monochrome interface. Text/icon distinguish it
   as well, so color is never the only signal. */
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
  border-left:4px solid #b77912!important;background:#fff9ed!important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
  border-left:4px solid #b42318!important;background:#fff3f2!important;
}
[data-testid="stAlert"] [data-testid="stAlertContentError"] p,
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) svg {color:#b42318!important;}
[data-testid="stAlert"] [data-testid="stAlertContentWarning"] strong {color:#8b570b!important;}
[data-testid="stButton"] button {min-height:3rem;border-radius:4px;font-size:clamp(1.12rem,1.05rem + .15vw,1.24rem);font-weight:640;box-shadow:none;}
[data-testid="stButton"] button [data-testid="stMarkdownContainer"] p {color:inherit!important;font-size:clamp(1.12rem,1.05rem + .15vw,1.24rem)!important;line-height:1.25!important;}
[data-testid="stButton"] button[kind="primary"] {background:var(--ink);border-color:var(--ink);color:#fff;}
[data-testid="stButton"] button[kind="primary"]:hover {background:#333333;border-color:#333333;}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-baseweb="select"]>div {
  background:#fff!important;border:1px solid var(--line-strong)!important;border-radius:4px!important;
  font-size:1.24rem!important;
}
[data-testid="stSelectbox"] .react-aria-ComboBox [role="group"] {
  background:#fff!important;border:1px solid var(--line-strong)!important;border-radius:4px!important;
}
[data-testid="stTextArea"] textarea::placeholder,[data-testid="stTextInput"] input::placeholder {
  color:#858585!important;opacity:1!important;
}
button:focus-visible,a:focus-visible,textarea:focus-visible,input:focus-visible {
  outline:2px solid var(--ink)!important;outline-offset:2px!important;
}
.masthead {border-top:3px solid var(--ink);padding-top:.7rem;margin-bottom:.4rem;}
.masthead .kicker {color:var(--ink);font-size:1.05rem;font-weight:720;}
.status-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));align-items:stretch;margin:1.35rem 0;
  border-top:1px solid var(--line-strong);border-bottom:1px solid var(--line-strong);background:#fff;}
.status-cell {display:flex;flex-direction:column;justify-content:center;min-height:6.1rem;padding:.85rem 1rem;border-right:1px solid var(--line);}
.status-cell:last-child {border-right:0;}
.status-label {display:block;font-size:1.1rem;color:var(--muted);margin-bottom:.3rem;}
.status-value {display:block;font-size:1.22rem;font-weight:700;color:var(--ink);line-height:1.4;}
.module-label {display:inline-block;font-size:1.16rem;letter-spacing:.02em;
  color:var(--muted);font-weight:700;}
.st-key-public_rag_module,.st-key-public_agent_module {
  position:relative;isolation:isolate;box-sizing:border-box;
  min-height:25rem;padding:4.2rem clamp(3.5rem,4.6vw,4.5rem) 3.7rem!important;
  background:var(--paper)!important;color:var(--ink);
  border:0!important;border-radius:0!important;box-shadow:none!important;
}
.st-key-public_rag_module::before,.st-key-public_agent_module::before {
  content:"";position:absolute;inset:0;z-index:0;pointer-events:none;
  background-repeat:no-repeat;background-position:center;background-size:100% 100%;
}
.st-key-public_rag_module::before {background-image:url("__RAG_FRAME__");}
.st-key-public_agent_module::before {background-image:url("__AGENT_FRAME__");}
.st-key-public_rag_module > *,.st-key-public_agent_module > * {position:relative;z-index:1;}
.st-key-public_rag_module > [data-testid="stElementContainer"]:last-child,
.st-key-public_agent_module > [data-testid="stElementContainer"]:last-child {margin-top:auto!important;}
.st-key-public_rag_module h3,.st-key-public_agent_module h3 {font-size:1.72rem!important;line-height:1.32;}
.st-key-public_rag_module [data-testid="stButton"],.st-key-public_agent_module [data-testid="stButton"] {margin-top:.15rem;}
@media (max-width:1200px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-public_rag_module) {flex-direction:column!important;align-items:stretch!important;}
  [data-testid="stHorizontalBlock"]:has(.st-key-public_rag_module) > [data-testid="stColumn"] {
    width:100%!important;max-width:100%!important;flex:1 1 100%!important;
  }
}
.module-detail {font-size:1.18rem;line-height:1.62;min-height:2.9rem;}
.flow-track,.review-steps {display:flex;gap:0;align-items:stretch;flex-wrap:wrap;margin:1rem 0;}
.flow-track {gap:.3rem 0;margin:.8rem 0 1rem;}
.flow-track span {display:inline-flex;align-items:center;color:var(--body);font-size:1.2rem;font-weight:620;}
.flow-track span:not(:last-child)::after {content:"→";color:var(--muted);padding:0 .8rem;font-weight:400;}
.review-steps {border-bottom:1px solid var(--line);gap:.25rem;margin:1.4rem 0;}
.review-steps span {padding:.55rem .7rem;border:0;border-bottom:2px solid transparent;
  color:var(--muted);font-size:1.16rem;font-weight:620;}
.review-steps .done {color:var(--body);}
.review-steps .current {color:var(--ink);border-bottom-color:var(--ink);font-weight:760;}
.agent-review-summary {border-left:3px solid var(--ink);margin:.55rem 0 1.1rem;padding:.55rem .95rem;background:var(--quiet);}
.agent-review-summary-heading {display:flex;align-items:baseline;justify-content:space-between;gap:1rem;color:var(--ink);font-size:1.08rem;}
.agent-review-summary-heading span {font-size:1rem;font-weight:700;white-space:nowrap;}
.agent-review-summary p {margin:.35rem 0 .1rem;line-height:1.65;color:var(--body);}
[class*="st-key-review_candidate_"] {border-top:1px solid var(--line-strong);padding:.75rem 0 .95rem;}
.agent-candidate-heading {display:flex;align-items:baseline;gap:.8rem;flex-wrap:wrap;color:var(--ink);font-size:1.25rem;line-height:1.4;}
.agent-candidate-heading span {font-size:1rem;font-weight:760;white-space:nowrap;}
.agent-candidate-heading strong {font-weight:750;}
.agent-field-label,.agent-evidence-heading {margin:.55rem 0 .15rem;color:var(--ink);font-size:1rem;font-weight:750;}
.agent-evidence-heading {border-left:2px solid var(--line-strong);padding-left:.6rem;margin-top:.9rem;}
.section-rule {border-top:1px solid var(--line-strong);padding-top:.72rem;margin:1.7rem 0 .85rem;
  color:var(--ink);font-size:1.16rem;font-weight:740;}
.context-strip {display:flex;gap:.5rem;flex-wrap:wrap;margin:.55rem 0 1.05rem;}
.context-strip span {background:#fff;border-right:1px solid var(--line);padding:.25rem .85rem .25rem 0;
  color:var(--body);font-size:1.1rem;}
.context-strip span:last-child {border-right:0;}
.context-strip strong {color:var(--ink);}
.notice-title {font-weight:750;color:var(--signal);font-size:1.08rem;}
.relation-confirmed {color:var(--ink);font-weight:730;}
.relation-suggested {color:var(--signal);font-weight:730;}
.mono {font-family:Consolas,"SFMono-Regular",monospace;font-size:1rem;}
.sidebar-mark {font-weight:770;font-size:1.3rem;color:var(--ink);border-bottom:2px solid var(--ink);
  padding-bottom:.65rem;margin-bottom:.55rem;}
.nav-heading {font-size:1.12rem;font-weight:740;color:var(--muted);padding:.8rem .3rem .25rem;
  border-top:1px solid var(--line);margin-top:.45rem;}
[class*="st-key-page_header_"] {overflow:visible;min-height:3rem;margin-bottom:.25rem;}
[class*="st-key-page_header_"] [data-testid="stHorizontalBlock"] {
  align-items:center;overflow:visible;gap:.35rem!important;width:max-content;max-width:100%;flex-wrap:wrap;
}
[class*="st-key-page_header_"] [data-testid="stColumn"] {
  flex:0 0 auto!important;width:auto!important;min-width:0!important;align-self:center!important;
}
[class*="st-key-page_header_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
  min-height:2.35rem;display:flex;align-items:center;justify-content:center;
}
[class*="st-key-page_header_"] [data-testid="stMarkdownContainer"] {overflow:visible;display:flex;align-items:center;min-height:2.35rem;margin:0!important;}
[class*="st-key-page_header_"] [data-testid="stMarkdownContainer"] p {margin:0!important;line-height:2.35rem!important;}
.breadcrumbs {font-size:1.2rem;line-height:1.45;color:var(--body);white-space:normal;overflow-wrap:anywhere;}
.breadcrumbs-current {height:2.35rem;min-height:2.35rem;box-sizing:border-box;display:flex;align-items:center;padding:0 .35rem;
  color:var(--ink);font-size:1.2rem;font-weight:740;line-height:1.35;overflow-wrap:anywhere;}
[class*="st-key-page_header_"] [class*="st-key-nav_back_"] button {
  width:auto!important;min-width:1.65rem!important;min-height:2.35rem!important;padding:0 .12rem!important;
  font-size:1.5rem!important;line-height:1!important;font-weight:700;
  background:transparent!important;color:var(--ink)!important;border:0!important;box-shadow:none!important;
}
[class*="st-key-page_header_"] [class*="st-key-nav_back_"] button:hover {
  background:transparent!important;color:var(--ink)!important;border:0!important;box-shadow:none!important;
}
[class*="st-key-page_header_"] [class*="st-key-breadcrumb_home_"] button,
[class*="st-key-page_header_"] [class*="st-key-breadcrumb_section_"] button {
  min-height:2.35rem;padding:.2rem .35rem;background:transparent!important;
  color:var(--body)!important;border:1px solid transparent!important;font-size:1.2rem;font-weight:620;
}
[class*="st-key-page_header_"] [class*="st-key-breadcrumb_home_"] button:hover,
[class*="st-key-page_header_"] [class*="st-key-breadcrumb_section_"] button:hover {
  background:transparent!important;color:var(--ink)!important;border-color:var(--line)!important;
}
.breadcrumb-separator {height:2.35rem;box-sizing:border-box;display:flex;align-items:center;justify-content:center;
  padding:0;color:var(--muted);font-size:1.1rem;line-height:1;text-align:center;}
[data-testid="stExpandSidebarButton"] {
  display:flex!important;visibility:visible!important;opacity:1!important;
  width:2.65rem!important;height:2.65rem!important;align-items:center;justify-content:center;
  background:var(--paper)!important;border:1px solid var(--line-strong)!important;
  border-radius:4px!important;color:var(--ink)!important;box-shadow:0 1px 3px rgba(0,0,0,.14)!important;
}
[data-testid="stSidebarCollapseButton"] {visibility:visible!important;opacity:1!important;}
[data-testid="stSidebarCollapseButton"] button {
  display:flex!important;visibility:visible!important;opacity:1!important;
  width:2.65rem!important;height:2.65rem!important;align-items:center;justify-content:center;
  background:var(--paper)!important;border:1px solid var(--line-strong)!important;
  border-radius:4px!important;color:var(--ink)!important;box-shadow:0 1px 3px rgba(0,0,0,.14)!important;
}
[data-testid="stExpandSidebarButton"] svg,[data-testid="stSidebarCollapseButton"] svg {
  width:1.3rem!important;height:1.3rem!important;color:var(--ink)!important;
}
.st-key-generated_answer {
  background:var(--quiet)!important;border:0!important;
  border-left:3px solid var(--ink)!important;border-radius:0!important;
}
.st-key-generated_answer [data-testid="stMarkdownContainer"] p {
  color:var(--ink)!important;font-size:1.3rem!important;line-height:1.75!important;
}
.st-key-generated_answer .citation-index {font-size:1.08rem!important;line-height:1.5!important;}
[class*="st-key-answer_claim_"] {max-width:76ch;}
[class*="st-key-answer_claim_"] + [class*="st-key-answer_claim_"] {
  border-top:1px solid var(--line);padding-top:1rem;
}
.evidence-preview {max-width:76ch;line-height:1.7;overflow-wrap:anywhere;color:var(--body);}
[class*="st-key-ocr_finding_"] {border-radius:4px!important;padding:1.1rem!important;}
[class*="st-key-ocr_finding_"] [data-testid="stMarkdownContainer"] p {
  max-width:76ch;line-height:1.7!important;
}
[class*="st-key-ocr_finding_"] [data-testid="stCode"] {max-height:20rem;overflow:auto;}
.st-key-knowledge_generate button,.st-key-knowledge_search button {
  background:var(--paper)!important;color:var(--ink)!important;
  border:1px solid var(--line-strong)!important;
}
.st-key-knowledge_generate button {font-weight:720;border-width:2px!important;}
.st-key-knowledge_search button {font-weight:620;color:var(--body)!important;}
.st-key-knowledge_generate button:hover,.st-key-knowledge_search button:hover {
  background:var(--paper)!important;color:var(--ink)!important;border-color:var(--ink)!important;
}
.st-key-knowledge_actions [data-testid="stHorizontalBlock"] {align-items:flex-start;}
#knowledge-request-anchor,#knowledge-results-anchor {scroll-margin-top:4.25rem;}
.st-key-knowledge_actions [data-testid="stPopover"] button {min-height:3rem;width:100%;}
@media (max-width:760px) {
  .st-key-knowledge_actions [data-testid="stHorizontalBlock"] {flex-wrap:wrap!important;gap:.5rem!important;}
  .st-key-knowledge_actions [data-testid="stColumn"] {min-width:9rem!important;flex:1 1 9rem!important;}
}
.citation-index {font-size:.98rem;color:var(--ink);font-weight:720;}
.st-key-knowledge_scope {border:0!important;border-top:1px solid var(--line)!important;
  border-bottom:1px solid var(--line)!important;border-radius:0!important;}
[class*="st-key-source_card_"],[class*="st-key-source_row_"],
[class*="st-key-limit_row_"],[class*="st-key-impact_row_"] {
  border:0!important;border-top:1px solid var(--line)!important;border-radius:0!important;
  box-shadow:none!important;
}
.st-key-before_panel,.st-key-after_panel {border:0!important;border-top:2px solid var(--ink)!important;
  border-bottom:1px solid var(--line)!important;border-radius:0!important;}
@media (max-width:760px) {.flow-track span:not(:last-child)::after {padding:0 .45rem;}}
@media (max-width:900px) {.status-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .status-cell:nth-child(2n) {border-right:0;}}
@media (max-width:600px) {.status-grid {grid-template-columns:minmax(0,1fr);}
  .status-cell,.status-cell:nth-child(2n) {border-right:0;border-bottom:1px solid var(--line);}
  .status-cell:last-child {border-bottom:0;}}
@media (max-width:760px) {.block-container {padding:4.5rem .85rem 2rem;}
  [class*="st-key-page_header_"] [data-testid="stHorizontalBlock"] {
    width:100%!important;max-width:100%!important;flex-wrap:wrap!important;
  }
  [class*="st-key-page_header_"] [data-testid="stColumn"] {
    flex:0 1 auto!important;max-width:100%!important;
  }
  .breadcrumbs-current {height:auto;min-height:2.35rem;}
  h1 {font-size:2rem!important;}.st-key-public_rag_module,.st-key-public_agent_module {
  min-height:23.5rem;padding:3.7rem 2.35rem 3.25rem!important;
  }
  .flow-track span {flex:0 1 auto;}.review-steps span {flex:1 1 42%;}}
</style>
"""

PUBLIC_CSS = (PUBLIC_CSS
              .replace("__RAG_FRAME__", _frame_data_uri("rag-book-frame.svg"))
              .replace("__AGENT_FRAME__", _frame_data_uri("agent-robot-frame.svg")))
