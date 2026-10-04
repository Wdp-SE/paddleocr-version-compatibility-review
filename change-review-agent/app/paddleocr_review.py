"""Controlled, read-only PaddleOCR application compatibility review workflow.

The static tool owns compatibility facts. This Agent checks request identity,
application locations and the pinned knowledge registry, then retrieves related
official material from the same gateway. It does not execute code or ask a model
to invent migration findings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol
from urllib.parse import urlsplit

from app.domain_profile import load_change_profile


PROJECT_ID = "paddleocr"
REPOSITORY = "PaddlePaddle/PaddleOCR"
SOURCE_VERSION = "v2.9.1"
TARGET_VERSION = "v3.0.0"
COMMITS = {
    SOURCE_VERSION: "07603421c20a96bb94bb87d0c4211032527ae706",
    TARGET_VERSION: "a8474288ad53c0f439c272b786c5fa6240f0cf27",
}
PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "paddleocr_change_profile.json"
STATUSES = {"supported_risk", "needs_verification", "unaffected"}


class PaddleOCRGateway(Protocol):
    def workspace(self) -> dict: ...
    def compatibility_review(self, files: list[dict], *, source_version: str, target_version: str) -> dict: ...
    def document(self, document_id: str) -> list[dict]: ...
    def search(self, question: str, *, version: str, language: str, top_k: int) -> dict: ...


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _version(value: object) -> str:
    # Requests use release tags. Evidence must use the registry's exact tag.
    return str(value or "")


def _request_files(files: object) -> dict[str, str]:
    if not isinstance(files, list) or not 1 <= len(files) <= 12:
        raise ValueError("请提供 1 到 12 个应用文本文件")
    submitted = {}
    total = 0
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "content"}:
            raise ValueError("应用文件只能包含 path 和 content")
        path, content = row["path"], row["content"]
        if (not isinstance(path, str) or not path or len(path) > 160
                or "\\" in path or ":" in path or "\x00" in path
                or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or PurePosixPath(path).suffix.lower() not in {".py", ".yaml", ".yml", ".txt", ".md", ".json"}
                or path.casefold() in {name.casefold() for name in submitted}):
            raise ValueError("应用文件路径必须是唯一的相对文本文件路径，不能包含路径穿越")
        if not isinstance(content, str):
            raise ValueError("应用文件 content 必须是文本")
        size = len(content.encode("utf-8"))
        total += size
        if size > 50_000 or total > 200_000 or len(content.splitlines()) > 5000:
            raise ValueError("应用文本超出审查限制")
        submitted[path] = content
    return submitted


def _workspace(workspace: object) -> dict:
    if not isinstance(workspace, dict):
        raise ValueError("未获取 PaddleOCR 知识空间")
    profile = workspace.get("domain_profile")
    versions = workspace.get("available_versions")
    if (workspace.get("workspace_id") != PROJECT_ID
            or not isinstance(profile, dict) or profile.get("id") != PROJECT_ID
            or workspace.get("repository") != REPOSITORY
            or workspace.get("repositories", [REPOSITORY]) != [REPOSITORY]
            or workspace.get("languages") != ["zh"]
            or not isinstance(versions, list) or not all(isinstance(version, str) for version in versions)
            or not set(COMMITS).issubset(set(versions))
            or workspace.get("rag_ready") is False):
        raise ValueError("兼容性审查需要匹配的 PaddleOCR 固定版本官方中文知识空间")
    return workspace


def _location(location: object, files: dict[str, str]) -> None:
    if not isinstance(location, dict) or location.get("path") not in files:
        raise ValueError("影响项未绑定提交的应用文件")
    lines = files[location["path"]].splitlines()
    start, end = location.get("line"), location.get("end_line")
    if (type(start) is not int or type(end) is not int
            or not 1 <= start <= end <= len(lines)):
        raise ValueError("影响项应用行号无效")
    snippet = location.get("snippet")
    if not isinstance(snippet, str) or not snippet.strip() or snippet not in "\n".join(lines[start - 1:end]):
        raise ValueError("影响项应用摘录与提交文本不一致")


def _source_value(row: dict, key: str):
    aliases = {"url": ("url", "source_url"), "path": ("path", "document_path"),
               "sha256": ("sha256", "source_sha256")}
    return next((row[name] for name in aliases.get(key, (key,)) if row.get(name) is not None), None)


def _registered_source(evidence: dict, registry: dict[str, dict]) -> dict:
    source = registry.get(evidence.get("source_id"))
    if not isinstance(source, dict) or evidence.get("version") not in COMMITS:
        raise ValueError("官方证据不属于已登记的固定版本来源")
    version = evidence["version"]
    if (source.get("repository") != REPOSITORY
            or _version(source.get("version")) != version
            or source.get("commit") != COMMITS[version]
            or any(_source_value(evidence, key) != _source_value(source, key)
                   for key in ("sha256", "commit", "url", "path"))
            or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256") or ""))):
        raise ValueError("官方证据版本、提交或文件哈希与知识空间不一致")
    parsed = urlsplit(str(evidence.get("url") or ""))
    if (parsed.scheme != "https" or parsed.netloc != "github.com"
            or parsed.path != f"/{REPOSITORY}/blob/{COMMITS[version]}/{evidence.get('path')}"
            or parsed.query or parsed.fragment):
        raise ValueError("官方证据 URL 必须指向登记的 GitHub 固定提交文件")
    if (type(evidence.get("line_start")) is not int or type(evidence.get("line_end")) is not int
            or not 1 <= evidence["line_start"] <= evidence["line_end"]
            or not isinstance(evidence.get("chunk_id"), str) or not evidence["chunk_id"]):
        raise ValueError("官方证据缺少有效片段与行号")
    return source


def _registered_chunk(evidence: dict, source: dict, chunks: list[dict]) -> dict:
    if not isinstance(chunks, list):
        raise ValueError("官方证据文档片段格式无效")
    chunk = next((row for row in chunks if isinstance(row, dict) and row.get("chunk_id") == evidence["chunk_id"]), None)
    if not isinstance(chunk, dict):
        raise ValueError("官方证据片段不在同一知识索引中")
    if (chunk.get("source_id") != evidence["source_id"] or chunk.get("version") != evidence["version"]
            or any(_source_value(chunk, key) != _source_value(source, key)
                   for key in ("sha256", "commit", "url", "path"))):
        raise ValueError("官方证据片段来源身份不一致")
    start = chunk.get("line_start", chunk.get("start_line"))
    end = chunk.get("line_end", chunk.get("end_line"))
    if type(start) is not int or type(end) is not int or not start <= evidence["line_start"] <= evidence["line_end"] <= end:
        raise ValueError("官方证据行号超出登记片段")
    return chunk


def _narrative(report: dict) -> str:
    labels = {"supported_risk": "静态风险有证据支持", "needs_verification": "需要补充验证",
              "unaffected": "基础调用形式未发现变更"}
    text = [f"PaddleOCR {report['source_version']} → {report['target_version']} 应用兼容性审查",
            f"静态结果：{labels[report['status']]}。实际 OCR 推理未验证。"]
    for finding in report["findings"]:
        app = finding["application"]
        text.append(f"- {app['path']}:{app['line']}：{finding['title']}。{finding['explanation']}")
    for gap in report.get("gaps") or []:
        text.append(f"- 待验证：{gap.get('detail', '')}")
    return "\n".join(text)


def _advice_summary(report: dict) -> str:
    """Bound optional model input; the complete static report stays authoritative."""
    prefix = (
        "仅针对已绑定官方证据的 PaddleOCR 升级静态发现整理人工核查建议，"
        "不要新增或改变静态发现，不得声称实际推理已验证。"
        "同一 evidence_chunk_id 的多个应用位置须合并为一项建议，动作中可概括多个位置。\n"
        f"固定升级：{report['source_version']} → {report['target_version']}；"
        f"发现 {len(report['findings'])} 项，缺口 {len(report.get('gaps') or [])} 项。\n"
    )
    notice = "\n摘要截断：模型只处理列出的核查点；全部发现和缺口仍以完整静态报告为准。"
    limit = 4000 - len(notice)
    lines = []
    for finding in report["findings"]:
        app = finding["application"]
        lines.append(f"{app['path']}:{app['line']}：{finding['title'][:120]}；"
                     f"{finding['explanation'][:180]}；核查：{finding.get('desired_check', '')[:150]}")
    for gap in report.get("gaps") or []:
        lines.append("待验证：" + str(gap.get("detail") or "")[:180])
    text = prefix
    for line in lines:
        if len(text) + len(line) + 1 > limit:
            return text + notice
        text += line + "\n"
    return text


def _validate_impact_extension(report, files, findings):
    if 'impact_schema_version' not in report:return
    if report['impact_schema_version']!=1:raise ValueError('unknown impact schema')
    graph=report.get('application_graph',{})
    nodes=graph.get('nodes');edges=graph.get('edges')
    if not isinstance(nodes,list) or len(nodes)>2000 or not isinstance(edges,list) or len(edges)>4000:
        raise ValueError('invalid application graph bounds')
    def position(loc):
        if not isinstance(loc,dict) or loc.get('path') not in files:raise ValueError('graph application path')
        text=files[loc['path']]; lines=text.splitlines(); lo,hi=loc.get('line_start'),loc.get('line_end')
        if (type(lo) is not int or type(hi) is not int or not 1<=lo<=hi<=len(lines)
                or loc.get('sha256')!=hashlib.sha256(text.encode()).hexdigest()
                or loc.get('excerpt')!='\n'.join(lines[lo-1:hi])):
            raise ValueError('graph application location mismatch')
    ids=set()
    for node in nodes:
        if not isinstance(node,dict) or node.get('id') in ids:raise ValueError('graph node identity')
        position(node.get('application'));ids.add(node['id'])
    for edge in edges:
        if edge.get('source') not in ids or edge.get('target') not in ids or edge.get('kind') not in ('reference','result_flow'):
            raise ValueError('graph edge identity')
        position(edge.get('application'))
    paths=report.get('impact_paths',[])
    findings_by_id={f['finding_id']:f for f in findings}
    if not isinstance(paths,list) or len(paths)>100:raise ValueError('impact path budget')
    for path in paths:
        chain=path.get('edges')
        if (path.get('finding_id') not in findings_by_id or path.get('level')!='candidate'
                or path.get('runtime_verified') is not False or not isinstance(chain,list)
                or not 1<=len(chain)<=8 or any(e not in edges for e in chain)
                or path.get('origin_node') not in ids or path.get('target_node') not in ids):
            raise ValueError('impact path proof level or identity mismatch')
        origin=next(n['application'] for n in nodes if n['id']==path['origin_node'])
        finding=findings_by_id[path['finding_id']]['application']
        start=finding.get('line_start',finding.get('line',1))
        if not (origin['path']==finding['path'] and origin['line_start']<=start<=origin['line_end']):
            raise ValueError('impact origin does not contain finding location')
        position(path.get('application'))
        if path['application']!=next(n['application'] for n in nodes if n['id']==path['target_node']):
            raise ValueError('impact target location mismatch')
        current=path['origin_node']
        for item in chain:
            source,target=(item['source'],item['target']) if item['kind']=='result_flow' else (item['target'],item['source'])
            if current!=source:raise ValueError('disconnected impact chain')
            current=target
        if current!=path['target_node']:raise ValueError('impact chain target mismatch')
        if path.get('flow_proven')!=all(e['kind']=='result_flow' for e in chain):
            raise ValueError('impact flow proof mismatch')


class PaddleOCRReviewAgent:
    def __init__(self, gateway: PaddleOCRGateway):
        self.gateway = gateway

    def analyze(self, files: list[dict], *, source_version: str = SOURCE_VERSION,
                target_version: str = TARGET_VERSION, generate_advice: bool = False,
                on_progress: Callable[[str], None] | None = None, resume:dict|None=None,
                cancelled=None, planner=None, investigation_progress=None) -> dict:
        import time
        from app.paddleocr_investigation import _bounded_method, _digest
        started=time.monotonic()
        previous_elapsed=(resume or {}).get('investigation',{}).get('elapsed_seconds',0)
        def remaining():return max(.001,240-previous_elapsed-(time.monotonic()-started))
        profile = load_change_profile(PROFILE_PATH)
        if (profile["id"] != PROJECT_ID or profile.get("repository") != REPOSITORY
                or profile.get("source_version") != SOURCE_VERSION or profile.get("target_version") != TARGET_VERSION):
            raise ValueError("PaddleOCR 审查配置与固定项目版本不匹配")
        if (source_version, target_version) != (SOURCE_VERSION, TARGET_VERSION):
            raise ValueError("本审查仅支持 PaddleOCR v2.9.1 升级到 v3.0.0")
        submitted = _request_files(files)
        workspace = _workspace(self.gateway.workspace())
        task_identity=_fingerprint({'files':files,'versions':[source_version,target_version],
                                    'corpus':workspace.get('corpus_fingerprint'),'generate_advice':generate_advice})
        if resume and resume.get('task_identity')!=task_identity:raise ValueError('输入、语料或设置已变化，不能恢复旧任务')
        if on_progress:
            on_progress("static_review")
        report = copy.deepcopy(self.gateway.compatibility_review(
            files, source_version=source_version, target_version=target_version,
            **({'request_timeout':remaining()} if _bounded_method(self.gateway.compatibility_review) else {})))
        if (not isinstance(report, dict) or report.get("schema_version") != 1
                or report.get("project_id") != PROJECT_ID
                or report.get("workspace_id", PROJECT_ID) != PROJECT_ID
                or report.get("source_version") != source_version
                or report.get("target_version") != target_version
                or report.get("runtime_verified") is not False
                or report.get("status") not in STATUSES):
            raise ValueError("静态工具返回的项目、版本或验证状态不匹配")
        file_rows = report.get("files")
        if (not isinstance(file_rows, list) or not all(isinstance(row, dict) for row in file_rows)
                or len(file_rows) != len(submitted)
                or {row.get("path") for row in file_rows} != set(submitted)):
            raise ValueError("静态工具未保留完整应用文件身份")
        for row in file_rows:
            content = submitted[row["path"]]
            if (row.get("sha256") != hashlib.sha256(content.encode("utf-8")).hexdigest()
                    or row.get("line_count") != len(content.splitlines())):
                raise ValueError("静态工具应用文件哈希或行数不一致")
        findings = report.get("findings")
        if not isinstance(findings, list) or len(findings) > 100:
            raise ValueError("静态工具影响清单格式无效")
        registry = {row.get("source_id"): row for row in workspace.get("source_registry") or [] if isinstance(row, dict)}
        chunks_by_source, checked, finding_ids = {}, {}, set()
        if on_progress:
            on_progress("evidence_validation")
        for finding in findings:
            if (not isinstance(finding, dict) or finding.get("status") not in STATUSES
                    or not isinstance(finding.get("finding_id"), str) or not finding["finding_id"]
                    or finding["finding_id"] in finding_ids
                    or not all(isinstance(finding.get(key), str) and finding[key].strip()
                               for key in ("rule_id", "title", "explanation", "desired_check"))):
                raise ValueError("静态工具影响项格式无效")
            finding_ids.add(finding["finding_id"])
            _location(finding.get("application"), submitted)
            evidence = finding.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("影响项必须绑定官方证据")
            versions = set()
            for citation in evidence:
                if not isinstance(citation, dict):
                    raise ValueError("官方证据格式无效")
                source = _registered_source(citation, registry)
                source_id = citation["source_id"]
                if source_id not in chunks_by_source:
                    chunks_by_source[source_id] = self.gateway.document(source.get("document_id") or source_id,
                        **({'request_timeout':remaining()} if _bounded_method(self.gateway.document) else {}))
                chunk = _registered_chunk(citation, source, chunks_by_source[source_id])
                checked[chunk["chunk_id"]] = chunk
                versions.add(citation["version"])
            if finding["status"] in {"supported_risk", "unaffected"} and versions != set(COMMITS):
                raise ValueError("静态兼容性结论必须绑定升级前后两版官方证据")
        gaps = report.get("gaps")
        if not isinstance(gaps, list) or len(gaps) > 100 or not all(isinstance(gap, dict) and isinstance(gap.get("code"), str)
                                                and isinstance(gap.get("detail"), str) for gap in gaps):
            raise ValueError("静态工具证据缺口格式无效")
        for gap in gaps:
            if gap.get("application") is not None:
                _location(gap["application"], submitted)
        steps = report.get("verification_steps")
        if not isinstance(steps, list) or not steps or not all(isinstance(step, str) and step.strip() for step in steps):
            raise ValueError("静态工具必须保留实际推理验证清单")
        expected_status = ("supported_risk" if any(f["status"] == "supported_risk" for f in findings)
                           else "needs_verification" if gaps or any(f["status"] == "needs_verification" for f in findings)
                           else "unaffected")
        if report["status"] != expected_status or report["status"] == "unaffected" and not findings:
            raise ValueError("静态工具总体状态与影响项、缺口不一致")
        expected_summary = {"finding_count": len(findings), "risk_count": sum(f["status"] == "supported_risk" for f in findings),
                            "gap_count": len(gaps)}
        if report.get("summary") != expected_summary:
            raise ValueError("静态工具报告计数与影响清单不一致")
        # Each check uses the same source registry as static findings.
        from app.paddleocr_investigation import build_checks, investigate, _bounded_method
        import time
        def validate_lookup(row, remaining):
            citation={key:_source_value(row,key) for key in ('source_id','chunk_id','version','sha256','commit','url','path')}
            citation.update(line_start=row.get('line_start'),line_end=row.get('line_end'))
            registered=_registered_source(citation,registry)
            source_id=citation['source_id']
            if source_id not in chunks_by_source:
                if remaining<=0 or not _bounded_method(self.gateway.document):
                    raise ValueError('DEADLINE_NOT_ENFORCEABLE')
                chunks_by_source[source_id]=self.gateway.document(registered.get('document_id') or source_id,request_timeout=remaining)
            original=_registered_chunk(citation,registered,chunks_by_source[source_id])
            if row.get('content')!=original.get('content'):
                raise ValueError('检索原文与登记来源不一致')
            return original
        _validate_impact_extension(report,submitted,findings)
        if on_progress:on_progress('official_lookup')
        checks=build_checks(report)
        restored=copy.deepcopy(resume['investigation']) if resume else {'identity':_digest(checks),'completed':{},'search_calls':0,'planner_calls':0,'elapsed_seconds':0}
        restored['elapsed_seconds']=min(240,previous_elapsed+time.monotonic()-started)
        if planner is None and generate_advice:planner=getattr(self.gateway,'plan_investigation',None)
        investigation=investigate(checks,self.gateway,clock=time.monotonic,validate_evidence=validate_lookup,
                                  resume=restored,cancelled=cancelled,
                                  planner=planner,on_progress=investigation_progress)
        related={row['chunk_id']:row for row in investigation['checked_evidence']}
        lookup_gaps=[f"{item['check_id']} / {version}：查证未完成（{item['reason']}，停止状态 {investigation['stop_reason']}）。"
                     for item in investigation['unresolved'] for version in item['versions']]
        model_calls, model_suggestions, model_status, model_generation = (resume or {}).get('advice_request_count',0), [], "NOT_REQUESTED", {}
        support_calls=(resume or {}).get('support_request_count',0)
        advice_cache=(resume or {}).get('advice_response')
        target_evidence = {key: row for key, row in {**related,**checked}.items() if row.get("version") == target_version}
        if generate_advice and not (cancelled and cancelled()) and investigation['elapsed_seconds']<240:
            advice_method = getattr(self.gateway, "review_advice_for_version", None)
            if not target_evidence:
                model_status = "NO_VERIFIED_TARGET_EVIDENCE"
            elif not callable(advice_method):
                model_status = "GENERATION_NOT_CONFIGURED"
            elif not _bounded_method(advice_method):
                model_status = 'DEADLINE_NOT_ENFORCEABLE'
            else:
                if on_progress:
                    on_progress("model_advice")
                allowed = dict(list(target_evidence.items())[:8])
                try:
                    if advice_cache is None:
                        if model_calls>=1:
                            raise RuntimeError('advice request budget already consumed')
                        model_calls+=1
                        advice = advice_method(_advice_summary(report),list(allowed),
                            version=target_version,request_timeout=remaining())
                        request_count=advice.get('claim_verification',{}).get('request_count')
                        support_calls=support_calls+request_count if type(request_count) is int and support_calls is not None else None
                        advice_cache=copy.deepcopy(advice)
                    else:advice=copy.deepcopy(advice_cache)
                    model_status = str(advice.get("status") or "INVALID_RESPONSE")
                    model_generation = advice.get("generation") if isinstance(advice.get("generation"), dict) else {}
                    review = advice.get("review") or {}
                    if model_status=='OK' and advice.get('claim_verification',{}).get('status') not in ('SUPPORTED','PARTIAL_SUPPORTED'):
                        model_status='SUPPORT_CHECK_NOT_PASSED'
                    if model_status == "OK" and isinstance(review, dict) and review.get("review_status") == "REQUIRES_HUMAN_REVIEW":
                        for candidate in review.get("impact_candidates") or []:
                            if (isinstance(candidate, dict) and candidate.get("evidence_chunk_id") in allowed
                                    and all(isinstance(candidate.get(key), str) and candidate[key].strip()
                                            for key in ("reason", "suggested_action"))):
                                model_suggestions.append({
                                    "reason": candidate["reason"], "suggested_action": candidate["suggested_action"],
                                    "evidence": allowed[candidate["evidence_chunk_id"]], "verified": False,
                                })
                        if not model_suggestions:
                            model_status = "ABSTAINED"
                    elif model_status == "OK":
                        model_status = "ABSTAINED"
                except Exception:
                    model_status = "GENERATION_PROVIDER_UNAVAILABLE"
                    advice_cache={'status':model_status,'claim_verification':{'status':'UNSUPPORTED','request_count':0}}
        request_identity = {"workspace_id": PROJECT_ID, "source_version": source_version,
                            "target_version": target_version, "files": file_rows,
                            "corpus_fingerprint": workspace.get("corpus_fingerprint")}
        fingerprint = _fingerprint(request_identity)
        total_elapsed=min(240,previous_elapsed+time.monotonic()-started)
        investigation['resume']['elapsed_seconds']=total_elapsed
        return {
            "schema_version": 1, "task_id": f"paddleocr-{fingerprint[:24]}",
            "request_fingerprint": fingerprint, "report_fingerprint": _fingerprint(report),
            "request_mode": "paddleocr_compatibility", "workspace_id": PROJECT_ID, "project_id": PROJECT_ID,
            "source_version": source_version, "target_version": target_version,
            "request_summary": f"PaddleOCR {source_version} → {target_version}：" + "、".join(submitted),
            "status": report["status"], "mode": "static", "runtime_verified": False,
            "model_calls": 0 if not model_calls and not investigation['planner_calls'] else None,
            "model_planning_request_count": investigation['planner_calls'],
            "model_support_request_count": support_calls, "task_elapsed_seconds":total_elapsed,
            "model_advice_request_count": model_calls, "model_status": model_status,
            "model_suggestions": model_suggestions, "model_generation": model_generation,
            "answer_source": "validated_static_tool", "compatibility_report": report,
            "rendered_report": _narrative(report), "retrieved_results": list(checked.values()),
            "related_materials": list(related.values()), "lookup_gaps": lookup_gaps,
            "investigation": investigation, "impact_schema_version":report.get("impact_schema_version"),
            "resume_state":{'task_identity':task_identity,'investigation':investigation['resume'],
                            'advice_request_count':model_calls,'support_request_count':support_calls,'advice_response':advice_cache},
            "evidence_gaps": [gap["detail"] for gap in gaps] + lookup_gaps,
            "stage_status": {"input_validation": "completed", "static_review": "completed",
                             "evidence_validation": "completed", "official_lookup": "partial" if lookup_gaps else "completed",
                             "model_advice": "completed" if model_suggestions else "skipped" if not generate_advice else "unavailable",
                             "human_review": "pending"},
            "public_baseline_written": False, "manual_review_required": True,
        }
