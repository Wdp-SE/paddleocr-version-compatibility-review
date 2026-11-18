"""Read-only, deliberately narrow configuration delta tool. Never imports code.

One directly imported PaddleOCR constructor is matched per snapshot. Its numeric
drop_score may be literal or follow a bounded, unambiguous declaration chain.
Findings describe submitted declarations, not runtime reachability or loading.
"""
from __future__ import annotations
import ast
import hashlib
import math
from src.paddleocr_config_values import DeclarationResolver,masked_declaration


def _inventory(files):
    if not isinstance(files, list) or not 1 <= len(files) <= 12:
        raise ValueError('需要 1–12 个文件')
    total, seen, rows = 0, set(), []
    for file in files:
        path, text = file['path'], file['content']
        if (not isinstance(path, str) or not path or path in seen or '\\' in path or ':' in path
                or path.startswith('/') or any(p in ('', '.', '..') for p in path.split('/'))
                or not isinstance(text, str) or not text or len(text.splitlines()) > 5000):
            raise ValueError('无效的应用文本或路径')
        raw = text.encode(); total += len(raw); seen.add(path)
        if len(raw) > 50000 or total > 200000:
            raise ValueError('应用文本超出预算')
        rows.append({'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'line_count': len(text.splitlines())})
    return rows


def _setting(files):
    candidates, gaps = [], []
    for file in files:
        if not file['path'].endswith('.py'):
            continue
        try:
            tree = ast.parse(file['content'])
        except (SyntaxError, ValueError, RecursionError):
            gaps.append({'code': 'unparsed_python', 'detail': file['path'] + ' 无法解析。'})
            continue
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        try:resolver=DeclarationResolver(files,file['path'],tree)
        except ValueError:
            gaps.append({'code':'configuration_resolution_budget','detail':file['path']+' 配置解析超出预算。'})
            continue
        def scope(node):
            parent = parents.get(id(node))
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return parent
                parent = parents.get(id(parent))
            return tree
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and n.module == 'paddleocr' and not n.level
                   for a in n.names if a.name == 'PaddleOCR']
        aliases = [a.asname or a.name for n in imports for a in n.names if a.name == 'PaddleOCR']
        for alias in set(aliases):
            # Reject all rebinding/shadowing and conditional execution rather than
            # reporting a guessed runtime value. Even a later assignment matters.
            shadow = any((isinstance(n, ast.Name) and n.id == alias and isinstance(n.ctx, (ast.Store, ast.Del)))
                         or (isinstance(n, ast.arg) and n.arg == alias)
                         or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == alias)
                         or (isinstance(n, (ast.Global, ast.Nonlocal)) and alias in n.names)
                         or (isinstance(n, ast.alias) and (n.asname or n.name) == alias and n not in
                             [a for imp in imports for a in imp.names if a.name == 'PaddleOCR'])
                         for n in ast.walk(tree))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == alias
                     and (scope(imports[0]) is tree or scope(n) is scope(imports[0]))]
            if len(aliases) != 1 or shadow:
                gaps.append({'code': 'constructor_binding_unknown', 'detail': file['path'] + ' 构造器绑定存在重定义或遮蔽。'})
                continue
            import_parent = parents.get(id(imports[0]))
            while import_parent is not None:
                if isinstance(import_parent, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)):
                    gaps.append({'code': 'conditional_import', 'detail': file['path'] + ' 的组件导入位于条件分支。'})
                    calls = []
                    break
                import_parent = parents.get(id(import_parent))
            for call in calls:
                ancestor, conditional = parents.get(id(call)), False
                while ancestor is not None:
                    if isinstance(ancestor, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match,
                                             ast.Lambda, ast.IfExp, ast.ClassDef, ast.AsyncFunctionDef)):
                        conditional = True
                    if isinstance(ancestor, ast.FunctionDef) and ancestor.decorator_list:
                        conditional = True
                    ancestor = parents.get(id(ancestor))
                options = [kw.value for kw in call.keywords if kw.arg == 'drop_score']
                valid = (not conditional and not call.args and len(options) == 1 and all(k.arg for k in call.keywords)
                         and (scope(imports[0]) is tree or call.lineno > imports[0].lineno))
                try:
                    value,sources=resolver.resolve(options[0]) if valid else (None,[])
                except (ValueError, TypeError, SyntaxError, RecursionError, KeyError):
                    value,sources = None,[]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    gaps.append({'code': 'effective_parameter_unknown', 'detail': file['path'] +
                                 ' 的 drop_score 不是可确认的直接数值配置；请补充配置解析与覆盖关系。'})
                    continue
                lines = file['content'].splitlines()
                shared=[loc for source in sources for loc in source.get('shared_consumers',[])]
                if shared:
                    gaps.append({'code':'shared_configuration_value','blocking':True,
                                 'detail':file['path']+' 的阈值来源还被其他位置消费；保留阈值变化，但需审查共享值的额外影响。',
                                 'related_locations':shared})
                candidates.append({'value': value, 'path': file['path'], 'line': call.lineno,
                                   'end_line': call.end_lineno, 'snippet': '\n'.join(lines[call.lineno-1:call.end_lineno]),
                                 'sha256': hashlib.sha256(file['content'].encode()).hexdigest(),
                                 'value_sources':sources})
    if len(candidates) != 1 or any(g['code']!='shared_configuration_value' for g in gaps):
        gaps.append({'code': 'configuration_scope_incomplete',
                     'detail': '每份快照需唯一且可解析的 PaddleOCR 显式 drop_score 调用；未证明动态配置或运行生效。'})
        return None, gaps
    return candidates[0], gaps


def review_configuration(before_files, after_files, *, dependency_version):
    if dependency_version != 'v2.9.1':
        raise ValueError('初版仅支持同一 v2.9.1 的 drop_score 配置变更')
    before_inventory, after_inventory = _inventory(before_files), _inventory(after_files)
    before, before_gaps = _setting(before_files)
    after, after_gaps = _setting(after_files)
    gaps = [{**g, 'snapshot': phase} for phase, rows in [('before', before_gaps), ('after', after_gaps)] for g in rows]
    old_files = {f['path']: f['content'] for f in before_files}
    new_files = {f['path']: f['content'] for f in after_files}
    if before and after:
        for path in old_files.keys() | new_files.keys():
            old_text, new_text = old_files.get(path), new_files.get(path)
            if old_text == new_text:
                continue
            try:
                relevant={before['path'],after['path']}|{s['path'] for loc in (before,after) for s in loc['value_sources']}
                only_threshold=(path in relevant and old_text is not None and new_text is not None
                    and masked_declaration(path,old_text,before)==masked_declaration(path,new_text,after))
            except (ValueError,TypeError,KeyError,SyntaxError,RecursionError):only_threshold=False
            if not only_threshold:
                gaps.append({'code': 'additional_code_changes', 'detail': path +
                             ' 还有阈值以外的变化；本流程不证明其他改动安全，请补充对应审查。'})
    changes = []
    if before and after:
        if before['path'] != after['path']:
            gaps.append({'code': 'entry_moved', 'detail': '构造器位置跨文件变化，初版不推断入口对应关系。'})
        elif before['value'] != after['value']:
            changes.append({'rule_id': 'ocr_drop_score_change', 'parameter': 'drop_score',
                            'before_value': before['value'], 'after_value': after['value'],
                            'direction': 'stricter_filter' if after['value'] > before['value'] else 'looser_filter',
                            'before': before, 'after': after,
                            'status': 'DECLARED_CHANGE_NOT_RUNTIME_VERIFIED'})
    return {'schema_version': 1, 'workflow_type': 'configuration_change', 'dependency_version': dependency_version,
            'status': 'NEEDS_CONTEXT' if gaps else 'DECLARED_CHANGE' if changes else 'NO_SUPPORTED_CHANGE',
            'before_files': before_inventory, 'files': after_inventory, 'changes': changes, 'gaps': gaps,
            'runtime_verified': False, 'scope': '受限构造器、常量、显式函数实参及已提交 JSON/YAML 声明链；不证明运行生效、推理质量或业务收益'}
