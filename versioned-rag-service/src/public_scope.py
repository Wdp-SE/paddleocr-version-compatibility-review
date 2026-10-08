"""Public-corpus scope guard for explicitly private organization requests."""

from __future__ import annotations

import re


_PRIVATE_ORG_CONTEXT = re.compile(
    r"(?:经纬恒润|本公司|公司|企业|组织|客户|本单位).{0,14}(?:内部|专属|私有|专有|自有)"
    r"|(?:内部|专属|私有|专有|自有).{0,10}(?:公司|企业|组织|客户|jira|工单|通讯录)",
    re.IGNORECASE,
)
_PRIVATE_DATA_SUBJECT = re.compile(
    r"(?:api|接口|jira|工单|审批|流程|制度|通讯录|手机号|授权名单|权限|密级|加密|车辆|质量|配置|映射|数据|"
    r"测试|通过率|回滚率|负责人|签字|验收|老化)",
    re.IGNORECASE,
)
_PRIVATE_ORG_ENGLISH = re.compile(
    r"\b(?:internal|private|proprietary|company[- ]specific|organization[- ]specific)"
    r"\s+(?:company\s+)?(?:api|endpoint|ticket|workflow|policy|directory|phone|permission|data)\b"
    r"|\b(?:our\s+)?company(?:'s)?\s+jira\s+(?:access[- ]approval|approver|admin|reviewer)"
    r"\s+(?:audit\s+trail|list|directory)\b"
    r"|\b(?:our\s+)?internal\s+(?:company\s+)?jira\s+(?:access[- ]approval|approver|admin|reviewer)"
    r"\s+(?:audit\s+trail|list|directory)\b"
    r"|\b(?:who|which\s+(?:person|team|group))\b.{0,48}\b(?:in|from|at)\s+(?:our|my)\s+"
    r"(?:company|organization|team)\b.{0,72}\b(?:approv\w*|jira|ticket|access[- ]request|permission)\b"
    r"|\b(?:find|show|retrieve|look\s+up|get|list)\b.{0,40}\b(?:our|my)\s+"
    r"(?:internal\s+)?(?:company(?:'s)?\s+)?jira\b.{0,48}\b(?:approv\w*|workflow|policy|"
    r"access[- ]request|permission|admin|reviewer|directory)\b"
    r"|\b(?:our|my)\s+(?:company|organization|team)(?:'s)?\b.{0,80}\b"
    r"(?:jira|ticket|approv\w*|workflow|access[- ]request|permission|directory|policy)\b",
    re.IGNORECASE,
)


def is_out_of_scope_public_request(text: str) -> bool:
    """Refuse requests that explicitly ask a public corpus for private company data."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return False
    private_request = bool((_PRIVATE_ORG_CONTEXT.search(normalized) and _PRIVATE_DATA_SUBJECT.search(normalized))
                           or _PRIVATE_ORG_ENGLISH.search(normalized))
    if private_request:
        return True
    # A checklist can be answered conditionally from component documentation;
    # it cannot establish that an unsubmitted application passed that checklist.
    checklist = re.search(r'需要(?:再|核查|检查|验证|提供|补充)|要核查哪些|还.{0,6}(?:资料|测试)|怎么(?:核查|验证)', normalized)
    direct_fact = re.search(r'请(?:直接)?(?:证明|保证|确认)|是否(?:已经|已)(?:通过|兼容)|(?:已经|已)通过(?:验收|回归)', normalized)
    return bool(
        (not checklist or direct_fact) and (re.search(r'我们(?:的)?应用|本应用|内部应用', normalized)
            and re.search(r'确认|保证|证明|通过|兼容', normalized)
            and re.search(r'输出|下游|升级|回归', normalized))
    )
