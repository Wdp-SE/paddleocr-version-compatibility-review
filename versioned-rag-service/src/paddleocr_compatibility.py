"""Bounded, evidence-bound static review of one pinned PaddleOCR migration.

Application paths are display labels. This module never imports, executes, rewrites,
or opens application code. ``unaffected`` describes only the observed basic API
surface; every report still requires real document inference and developer review.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from collections import Counter, deque
from pathlib import PurePosixPath


MAX_FILES = 12
MAX_PATH_CHARS = 160
MAX_FILE_BYTES = 50_000
MAX_TOTAL_BYTES = 200_000
MAX_LINES = 5_000
MAX_AST_NODES = 15_000
MAX_NESTING = 128
MAX_FINDINGS = 100
MAX_GAPS = 100
COMMITS = {
    "v2.9.1": "07603421c20a96bb94bb87d0c4211032527ae706",
    "v3.0.0": "a8474288ad53c0f439c272b786c5fa6240f0cf27",
}
REPOSITORY = "PaddlePaddle/PaddleOCR"
_DEVICE = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
_UNKNOWN = ("unknown",)


def _version(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("unsupported PaddleOCR version pair")
    return value if value.startswith("v") else "v" + value


def _files(files: list[dict]) -> list[dict]:
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("application file count must be between 1 and 12")
    result, seen, total = [], set(), 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "content"}:
            raise ValueError("each application file requires only path and content")
        path, content = item["path"], item["content"]
        if (
            not isinstance(path, str) or not 1 <= len(path) <= MAX_PATH_CHARS
            or path != path.strip() or "\\" in path or ":" in path
            or any(ord(c) < 32 for c in path) or path.startswith("/")
            or any(part in {"", ".", ".."} or part.endswith((" ", ".")) or _DEVICE.match(part) for part in path.split("/"))
        ):
            raise ValueError("application path must be a safe relative display label")
        if path.casefold() in seen:
            raise ValueError("duplicate application path")
        seen.add(path.casefold())
        if not isinstance(content, str) or not content or len(content) > MAX_FILE_BYTES:
            raise ValueError("application content exceeds allowed bytes or is empty")
        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("application content must be valid UTF-8 text") from None
        total += len(raw)
        if len(raw) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("application content exceeds allowed bytes")
        lines = content.splitlines()
        if len(lines) > MAX_LINES:
            raise ValueError("application content exceeds allowed lines")
        result.append({"path": path, "content": content, "lines": lines, "sha256": hashlib.sha256(raw).hexdigest()})
    return result


def _tree(content: str) -> ast.AST:
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(content).readline):
            if token.type == tokenize.OP:
                if token.string in ("(", "[", "{"):
                    depth += 1
                    if depth > MAX_NESTING:
                        raise ValueError("application parser nesting limit exceeded")
                elif token.string in (")", "]", "}"):
                    depth -= 1
    except (tokenize.TokenError, IndentationError):
        pass  # ast.parse below produces the non-sensitive syntax diagnostic.
    try:
        tree = ast.parse(content)
    except (RecursionError, MemoryError):
        raise ValueError("application AST resource limit exceeded") from None
    queue, count = deque([(tree, 0)]), 0
    while queue:
        node, depth = queue.popleft()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_NESTING:
            raise ValueError("application AST resource limit exceeded")
        queue.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


class _Evidence:
    """Use only chunks registered to the same fixed official source identity."""

    def __init__(self, index):
        manifest = getattr(index, "manifest", None)
        if (
            not isinstance(manifest, dict) or manifest.get("workspace_id") != "paddleocr"
            or manifest.get("project_id") not in {None, "paddleocr"} or manifest.get("repository") != REPOSITORY
        ):
            raise ValueError("compatibility review requires the PaddleOCR project index")
        self.chunks = []
        self.cache = {}
        sources = {source.get("source_id"): source for source in manifest.get("sources", []) if isinstance(source, dict)}
        for chunk in getattr(index, "chunks", []):
            if not isinstance(chunk, dict):
                continue
            source = sources.get(chunk.get("source_id"))
            if not source:
                continue
            version = source.get("version")
            path = source.get("document_path") or source.get("path")
            commit = COMMITS.get(version)
            digest = source.get("sha256")
            url = source.get("source_url")
            if (
                not commit or source.get("commit") != commit or source.get("repository") != REPOSITORY
                or not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                or url != f"https://github.com/{REPOSITORY}/blob/{commit}/{path}"
                or chunk.get("version") != version or chunk.get("commit") != commit
                or chunk.get("source_sha256") != digest or chunk.get("document_path") != path
                or chunk.get("repository") != REPOSITORY or not isinstance(chunk.get("content"), str)
                or not isinstance(chunk.get("chunk_id"), str)
                or type(chunk.get("line_start")) is not int or type(chunk.get("line_end")) is not int
                or not 1 <= chunk["line_start"] <= chunk["line_end"]
            ):
                continue
            self.chunks.append((chunk, source))

    def pick(self, version, path_test, *tokens):
        for chunk, source in self.chunks:
            if chunk["version"] == version and path_test(chunk["document_path"]) and all(token in chunk["content"] for token in tokens):
                return {
                    "source_id": source["source_id"], "chunk_id": chunk["chunk_id"], "version": version,
                    "sha256": source["sha256"], "commit": source["commit"], "url": source["source_url"],
                    "path": chunk["document_path"], "line_start": chunk["line_start"], "line_end": chunk["line_end"],
                    "title": chunk.get("document_title") or source.get("document_title") or chunk["document_path"],
                }
        return None

    def facts(self, rule):
        if rule in self.cache:
            return self.cache[rule]
        old, new = "v2.9.1", "v3.0.0"
        old_code = lambda p: p == "paddleocr.py"
        new_code = lambda p: p == "paddleocr/_pipelines/ocr.py"
        doc = lambda p: p.endswith(".md")
        if rule == "removed_ppstructure":
            rows = [
                self.pick(old, old_code, "class PPStructure("),
                self.pick(new, lambda p: "upgrade_notes" in p, "PPStructure", "PPStructureV3", "移除"),
            ]
        elif rule == "legacy_ocr_result":
            rows = [
                self.pick(old, old_code, "zip(dt_boxes, rec_res)", "ocr_res.append"),
                self.pick(new, new_code, "def ocr(", "self.predict"),
                self.pick(new, lambda p: p == "docs/version3.x/pipeline_usage/OCR.md", "rec_texts", "rec_scores"),
            ]
        else:
            rows = [
                self.pick(old, lambda p: p == "doc/doc_ch/whl.md", "PaddleOCR(", 'lang="ch"'),
                self.pick(old, old_code, "def ocr("),
                self.pick(new, new_code, "lang=None"),
                self.pick(new, new_code, 'lang == "ch"'),
                self.pick(new, new_code, "def ocr(", "self.predict"),
            ]
        if any(row is None for row in rows):
            self.cache[rule] = []
        else:
            self.cache[rule] = list({row["chunk_id"]: row for row in rows}.values())
        return self.cache[rule]


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _module_mutations(statements):
    """Pre-detect mutable global identities without executing module control flow."""
    counts, attribute_writes, forced = Counter(), set(), set()
    queue = deque(statements)
    while queue:
        node = queue.popleft()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] += 1
            globals_declared = {name for child in ast.walk(node) if isinstance(child, ast.Global) for name in child.names}
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)) and child.id in globals_declared:
                    forced.add(child.id)
                elif isinstance(child, ast.Attribute) and isinstance(child.ctx, (ast.Store, ast.Del)):
                    key = _name(child)
                    if key:
                        forced.add(key)
            continue  # Ordinary function/class local bindings are not module bindings.
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                counts[alias.asname or alias.name.split(".")[0]] += 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            counts[node.id] += 1
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            key = _name(node)
            if key:
                counts[key] += 1
                attribute_writes.add(key)
        queue.extend(ast.iter_child_nodes(node))
    return {key for key, count in counts.items() if count > 1} | forced, attribute_writes


class _Analyzer:
    def __init__(self, file, evidence, report):
        self.file, self.evidence, self.report = file, evidence, report
        self.related_names = set()
        self.seen = set()
        self.ocr_seen = False
        self.module_mutations = set()
        self.module_attribute_writes = set()

    def prepare(self, statements):
        self.module_mutations, self.module_attribute_writes = _module_mutations(statements)

    def bind(self, key, value, env):
        # A new parent object cannot inherit attributes of the former object.
        for child in [name for name in env if name.startswith(key + ".")]:
            del env[child]
        env[key] = value
        if value[0] != "unknown":
            self.related_names.add(key.split(".")[0])

    def deferred_env(self, node, env, local_names):
        reads = {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        mutations = self.module_mutations | {
            key for key in self.module_attribute_writes
            if env.get(key.split(".")[0], _UNKNOWN)[0] == "module"
        }
        for key in mutations:
            root = key.split(".")[0]
            if root in local_names or root not in reads:
                continue
            known = any(value[0] != "unknown" and (name == key or name.startswith(key + ".")) for name, value in env.items())
            known = known or ("." in key and env.get(root, _UNKNOWN)[0] == "module")
            if known:
                self.gap("deferred_global_binding", "函数或延迟表达式读取的全局 PaddleOCR 相关名称可能重新绑定；无法按定义位置证明调用时的目标。", node)
            self.bind(key, _UNKNOWN, env)
        return env

    def location(self, node):
        line = getattr(node, "lineno", 1)
        end = min(getattr(node, "end_lineno", line), line + 3, len(self.file["lines"]))
        return {"path": self.file["path"], "line": line, "end_line": end, "snippet": "\n".join(self.file["lines"][line - 1:end])[:600]}

    def gap(self, code, detail, node):
        key = ("gap", code, getattr(node, "lineno", 1))
        if key not in self.seen:
            self.seen.add(key)
            if len(self.report["gaps"]) >= MAX_GAPS - 1:
                self.overflow(node)
                return
            self.report["gaps"].append({"code": code, "detail": detail, "application": self.location(node)})

    def overflow(self, node):
        if not any(gap["code"] == "report_limit" for gap in self.report["gaps"]):
            self.report["gaps"].append({"code": "report_limit", "detail": "报告条目达到上限；其余位置未完整列出，需拆分输入并人工审核。", "application": self.location(node)})

    def finding(self, rule, status, title, explanation, check, node):
        key = ("finding", rule, getattr(node, "lineno", 1))
        if key in self.seen:
            return
        self.seen.add(key)
        if len(self.report["findings"]) >= MAX_FINDINGS:
            self.overflow(node)
            return
        evidence = self.evidence.facts(rule)
        if not evidence:
            self.gap("missing_official_evidence", f"{title}：固定版本官方证据不完整，无法生成支持结论。", node)
            return
        self.report["findings"].append({
            "finding_id": f"finding-{len(self.report['findings']) + 1}", "rule_id": rule, "status": status,
            "title": title, "explanation": explanation, "application": self.location(node),
            "evidence": evidence, "desired_check": check,
        })

    def value(self, node, env):
        key = _name(node)
        if key and key in env:
            return env[key]
        if isinstance(node, ast.NamedExpr):
            value = self.value(node.value, env)
            self.assign(node.target, value, env)
            return value
        if isinstance(node, ast.Lambda):
            local = env.copy()
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
            for arg in args:
                self.bind(arg.arg, _UNKNOWN, local)
            self.deferred_env(node, local, {arg.arg for arg in args})
            self.value(node.body, local)
            return _UNKNOWN
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            local = env.copy()
            for generator in node.generators:
                iterable = self.value(generator.iter, local)
                item = {"ocr_results": ("ocr_page",), "ocr_page": ("ocr_line",), "structure_results": ("structure_region",)}.get(iterable[0], _UNKNOWN)
                self.assign(generator.target, item, local)
                for condition in generator.ifs:
                    self.value(condition, local)
            for field in ("elt", "key", "value"):
                child = getattr(node, field, None)
                if child is not None:
                    self.value(child, local)
            return _UNKNOWN
        if isinstance(node, ast.Attribute):
            parent = self.value(node.value, env)
            if parent[0] == "module" and node.attr in {"PaddleOCR", "PPStructure"}:
                return ("api", node.attr)
            if parent[0] == "instance":
                return ("method", parent[1], node.attr, parent[2])
        if isinstance(node, ast.Subscript):
            parent = self.value(node.value, env)
            numeric = isinstance(node.slice, ast.Constant) and type(node.slice.value) is int
            if parent[0] == "ocr_results" and numeric:
                return ("ocr_page",)
            if parent[0] in {"ocr_page", "ocr_line", "legacy_value"} and numeric:
                self.finding(
                    "legacy_ocr_result", "supported_risk", "旧版 OCR 结果按嵌套列表位置消费",
                    "应用对 OCR 页/文本行使用整数位置访问。2.9.1 的 ocr() 组装框与识别结果的嵌套列表；3.0.0 的 ocr() 委托 predict()，官方结果示例使用 rec_texts、rec_scores 等字段。该旧消费契约需要迁移验证。",
                    "为现有归一化逻辑准备带文本、空页、多页的文档；核对目标结果对象及 rec_texts/rec_scores，逐项比较字段、类型、顺序和置信度。", node,
                )
                return ("legacy_value",)
            if parent[0] in {"structure_results", "structure_region"}:
                if isinstance(node.slice, ast.Constant) and node.slice.value in {"res", "type", "bbox"}:
                    self.finding(
                        "removed_ppstructure", "supported_risk", "旧版版面结果字典消费依赖已移除 API",
                        "应用消费 PPStructure 返回的区域字典字段。官方 3.0 升级说明移除 PPStructure 并由 PPStructureV3 替代；该处需要重新核对目标版面结果契约。",
                        "人工核对 PPStructureV3 的区域、表格和 OCR 输出，验证现有 res/type/bbox 映射及下游字段。", node,
                    )
                return ("structure_region",)
            self.value(node.slice, env)
            return _UNKNOWN
        if isinstance(node, ast.Call):
            function = self.value(node.func, env)
            for argument in node.args:
                self.value(argument, env)
            for keyword in node.keywords:
                self.value(keyword.value, env)
            if function[0] == "api":
                self.ocr_seen = True
                api = function[1]
                if api == "PPStructure":
                    self.finding(
                        "removed_ppstructure", "supported_risk", "PPStructure API 在目标版本移除",
                        "应用构造 PPStructure。固定版本 3.0 官方升级说明明确移除 PPStructure，并由 PPStructureV3 替代；需要人工调整版面解析适配器。",
                        "确认目标 PPStructureV3 初始化、输入及结果字段，并使用含表格、段落和图片区域的文档验证归一化输出。", node,
                    )
                dynamic = any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords)
                if dynamic:
                    self.gap("dynamic_arguments", "构造参数通过动态展开提供，无法静态核对。", node)
                allowed = not node.args and not dynamic and all(
                    k.arg == "lang" and isinstance(k.value, ast.Constant) and k.value.value == "ch" for k in node.keywords
                )
                if api == "PaddleOCR" and not allowed and not dynamic:
                    self.gap("unreviewed_parameter", "该构造参数不在已核对的基础调用范围；不能据此断言参数被移除或仍兼容。", node)
                return ("instance", api, allowed)
            if function[0] == "method":
                _, api, method, supported = function
                self.ocr_seen = True
                if api == "PaddleOCR" and method == "ocr":
                    if any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords):
                        self.gap("dynamic_arguments", "OCR 调用参数动态展开，需人工核对目标签名。", node)
                    elif supported and len(node.args) == 1 and not node.keywords:
                        self.finding(
                            "basic_ocr_surface", "unaffected", "基础 PaddleOCR(lang='ch').ocr(input) 调用表面有官方支持",
                            "已核对两版官方材料中的 PaddleOCR、lang 和 ocr(input) 入口；3.0.0 仍保留 ocr() 并委托 predict()。这里只支持观察到的调用表面，不证明输出消费、模型、环境或实际文档推理兼容。",
                            "在目标 PaddlePaddle/PaddleOCR 环境执行真实文档推理，核对模型加载、文本质量、空页、多页和下游归一化结果。", node,
                        )
                    else:
                        self.gap("unreviewed_call_arguments", "OCR 调用形态或构造配置超出已核对基础范围，需比对目标签名。", node)
                    return ("ocr_results",)
                if api == "PPStructure" and method == "__call__":
                    return ("structure_results",)
                self.gap("unreviewed_method", "PaddleOCR 对象方法超出当前已核对规则。", node)
                return _UNKNOWN
            if function[0] == "instance":
                if function[1] == "PPStructure":
                    return ("structure_results",)
                self.gap("unreviewed_method", "PaddleOCR 对象通过未核对调用方式使用。", node)
            if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "eval", "exec", "__import__"}:
                self.gap("dynamic_access", "动态访问或执行无法由静态审查确定 PaddleOCR 目标，待审文本未执行。", node)
            elif isinstance(node.func, ast.Call):
                self.gap("dynamic_access", "调用目标由另一个调用动态生成，无法静态确定。", node)
            elif key is None and (
                (isinstance(node.func, ast.Attribute) and node.func.attr in {"ocr", "predict", "PPStructure", "PaddleOCR"})
                or (_name(node.func) or "").split(".")[0] in self.related_names
            ):
                self.gap("unresolved_call", "可能的 PaddleOCR 调用无法解析为已登记导入或实例；需人工确认绑定。", node)
            return _UNKNOWN
        for child in ast.iter_child_nodes(node):
            self.value(child, env)
        return _UNKNOWN

    def assign(self, target, value, env):
        key = _name(target)
        if key:
            self.bind(key, value, env)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if value[0] in {"ocr_page", "ocr_line", "legacy_value"}:
                self.finding(
                    "legacy_ocr_result", "supported_risk", "旧版 OCR 结果按嵌套位置解包",
                    "应用通过元组或列表目标解包 OCR 页/文本行。2.9.1 的嵌套框与识别结果列表支持该旧消费方式；3.0.0 的 ocr() 委托 predict()，官方结果使用 rec_texts、rec_scores 等字段，需要重新核对解包契约。",
                    "使用真实文档验证框、文本与置信度的归一化映射；核对目标结果字段，避免继续按旧位置解包。", target,
                )
            item = ("ocr_page",) if value[0] == "ocr_results" else _UNKNOWN
            for element in target.elts:
                if isinstance(element, ast.Starred):
                    # Starred unpacking collects a list of pages, not one page.
                    rest = value if value[0] == "ocr_results" else _UNKNOWN
                    self.assign(element.value, rest, env)
                else:
                    self.assign(element, item, env)

    def block(self, statements, env):
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    self.bind(name, ("module",) if alias.name == "paddleocr" else _UNKNOWN, env)
                    if alias.name == "paddleocr":
                        self.related_names.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if node.module == "paddleocr" and not node.level:
                        self.related_names.add(name)
                        if name == "*":
                            self.gap("wildcard_import", "星号导入无法可靠解析 PaddleOCR 符号。", node)
                        self.bind(name, ("api", alias.name) if alias.name in {"PPStructure", "PaddleOCR"} else _UNKNOWN, env)
                    else:
                        self.bind(name, _UNKNOWN, env)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.value(node.value, env) if node.value else _UNKNOWN
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign):
                self.value(node.value, env)
                self.assign(node.target, _UNKNOWN, env)
            elif isinstance(node, ast.Expr):
                self.value(node.value, env)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    self.value(decorator, env)
                local = env.copy()
                args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
                local_names = {arg.arg for arg in args}
                for arg in args:
                    self.bind(arg.arg, _UNKNOWN, local)
                # Python treats all assigned/imported names as local throughout a function.
                for nested in ast.walk(node):
                    if isinstance(nested, ast.Name) and isinstance(nested.ctx, ast.Store):
                        local_names.add(nested.id)
                        self.bind(nested.id, _UNKNOWN, local)
                    elif isinstance(nested, (ast.Import, ast.ImportFrom)):
                        for alias in nested.names:
                            name = alias.asname or alias.name.split(".")[0]
                            local_names.add(name)
                            self.bind(name, _UNKNOWN, local)
                self.deferred_env(node, local, local_names)
                self.bind(node.name, _UNKNOWN, env)
                self.block(node.body, local)
            elif isinstance(node, ast.ClassDef):
                self.bind(node.name, _UNKNOWN, env)
                self.block(node.body, env.copy())
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                iterable = self.value(node.iter, env)
                local = env.copy()
                item = {"ocr_results": ("ocr_page",), "ocr_page": ("ocr_line",), "structure_results": ("structure_region",)}.get(iterable[0], _UNKNOWN)
                self.assign(node.target, item, local)
                self.block(node.body, local)
                self.block(node.orelse, local)
                self.invalidate(node, env)
            elif isinstance(node, (ast.If, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
                self.gap("uncertain_control_flow", "条件、异常或上下文控制流未做运行路径证明，相关绑定需人工确认。", node)
                self.invalidate(node, env)
                # Analyze possible bodies conservatively after invalidating their assignments.
                for field in ("body", "orelse", "finalbody"):
                    self.block(getattr(node, field, []), env.copy())
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        self.block(handler.body, env.copy())
                if isinstance(node, ast.Match):
                    for case in node.cases:
                        self.block(case.body, env.copy())
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    self.assign(target, _UNKNOWN, env)
            else:
                self.value(node, env)

    def invalidate(self, node, env):
        for nested in ast.walk(node):
            if isinstance(nested, ast.Name) and isinstance(nested.ctx, ast.Store):
                self.bind(nested.id, _UNKNOWN, env)
            elif isinstance(nested, ast.Attribute) and isinstance(nested.ctx, ast.Store):
                self.assign(nested, _UNKNOWN, env)


def review_compatibility(index, *, source_version: str, target_version: str, files: list[dict]) -> dict:
    """Return a JSON report for pasted application files; raise ValueError on bad input."""
    source_version, target_version = _version(source_version), _version(target_version)
    if (source_version, target_version) != ("v2.9.1", "v3.0.0"):
        raise ValueError("unsupported PaddleOCR version pair; expected v2.9.1 to v3.0.0")
    evidence, application_files = _Evidence(index), _files(files)
    report = {
        "schema_version": 1, "project_id": "paddleocr", "workspace_id": "paddleocr",
        "source_version": source_version, "target_version": target_version,
        "status": "needs_verification", "runtime_verified": False, "model_call_count": 0,
        "files": [{"path": f["path"], "sha256": f["sha256"], "line_count": len(f["lines"])} for f in application_files],
        "findings": [], "gaps": [],
        "verification_steps": [
            "开发者审核每个应用位置与固定版本官方证据，确认升级调整。",
            "在目标依赖环境执行真实扫描文档推理；核对模型加载、文本/版面质量、空页、多页和下游结果契约。静态审查不等于实际推理验证。",
        ],
    }
    for file in application_files:
        analyzer = _Analyzer(file, evidence, report)
        if PurePosixPath(file["path"]).suffix.casefold() != ".py":
            node = ast.Constant(value=None)
            node.lineno = node.end_lineno = 1
            matches = re.findall(r"(?im)\bpaddleocr\s*==\s*(v?\d+\.\d+\.\d+)\b", file["content"])
            if any(_version(match) != source_version for match in matches):
                analyzer.gap("dependency_version_mismatch", "配置声明的 PaddleOCR 依赖与审查源版本不一致。", node)
            analyzer.gap("unreviewed_configuration", "配置仅作为受限文本检查依赖声明；未加载 YAML 对象或证明完整配置迁移兼容。", node)
            continue
        try:
            tree = _tree(file["content"])
        except SyntaxError as error:
            node = ast.Constant(value=None)
            node.lineno = node.end_lineno = min(max(error.lineno or 1, 1), len(file["lines"]))
            analyzer.gap("syntax_error", "应用文本无法解析为 Python AST；未执行代码。", node)
            continue
        analyzer.prepare(tree.body)
        analyzer.block(tree.body, {})
        if not analyzer.ocr_seen:
            analyzer.gap("no_resolved_paddleocr_usage", "未识别到可证明的 PaddleOCR 构造及调用；空发现不能作为兼容证明。", tree)
    from src.paddleocr_application_graph import analyze_application, trace_impacts
    graph=analyze_application(files)
    report.update(impact_schema_version=1,application_graph=graph,
                  impact_paths=trace_impacts(graph,report['findings']))
    graph_gaps=[]
    for gap in graph['gaps']:
        item=dict(gap)
        if item.get('application'):
            loc=item['application']
            item['application']={'path':loc['path'],'line':loc['line_start'],
                                 'end_line':min(loc['line_end'],loc['line_start']+3),
                                 'snippet':'\n'.join(loc['excerpt'].splitlines()[:4])[:600]}
        graph_gaps.append(item)
    report['gaps']=(report['gaps']+graph_gaps)[:MAX_GAPS]
    report['regression_requirements']=[{
        'finding_id':finding['finding_id'],'application':finding['application'],
        'input_conditions':['带文本页','空页','多页'],
        'expected_contract':finding['desired_check'],'status':'NOT_EXECUTED',
        'runtime_verified':False} for finding in report['findings']]
    risks = sum(f["status"] == "supported_risk" for f in report["findings"])
    if risks:
        report["status"] = "supported_risk"
    elif report["gaps"] or not report["findings"]:
        report["status"] = "needs_verification"
    else:
        report["status"] = "unaffected"
    report["summary"] = {"finding_count": len(report["findings"]), "risk_count": risks, "gap_count": len(report["gaps"])}
    return report
