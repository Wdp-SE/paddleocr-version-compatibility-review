"""Bounded call-site argument propagation over submitted Python text. Never imports it."""
import ast

from src.paddleocr_function_results import local_module, import_target


class ArgumentFlow:
    max_depth = 6
    max_contexts = 200

    def __init__(self, files, evidence, report, summaries):
        from src.paddleocr_compatibility import _tree
        self.evidence, self.report, self.summaries = evidence, report, summaries
        self.functions, self.trees, self.files = {}, {}, {}
        self.active, self.cache, self.calls = [], {}, 0
        self.visited_paths = set()
        for file in files:
            if not file['path'].endswith('.py'):
                continue
            try:
                tree = _tree(file['content'])
            except (SyntaxError, ValueError):
                continue
            module = local_module(file['path'])
            self.trees[module], self.files[module] = tree, file
            counts = {}
            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    counts[statement.name] = counts.get(statement.name, 0) + 1
            for node in tree.body:
                if (not isinstance(node, ast.FunctionDef) or node.decorator_list or counts[node.name] != 1):
                    continue
                # Module rebinding and nested mutable state invalidate a static call target.
                outside = [n for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                           for n in ast.walk(s)]
                if any(isinstance(n, ast.Name) and n.id == node.name and isinstance(n.ctx, (ast.Store, ast.Del)) for n in outside):
                    continue
                if any(isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)) for n in ast.walk(node)):
                    continue
                self.functions[(module, node.name)] = (file, node)

    def bindings(self, file):
        module = local_module(file['path'])
        tree = self.trees.get(module)
        imported, own = {}, {}
        if tree is None:
            return imported, own
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = import_target(file['path'], node)
                for alias in node.names:
                    if (target, alias.name) in self.functions:
                        imported[(node.level, node.module, alias.name)] = ('local_function', target, alias.name)
            elif isinstance(node, ast.FunctionDef) and (module, node.name) in self.functions:
                own[node.name] = ('local_function', module, node.name)
        return imported, own

    def call(self, function, node, analyzer, positional, keywords):
        from src.paddleocr_compatibility import _Analyzer, _UNKNOWN
        key = tuple(function[1:])
        entry = self.functions.get(key)
        if entry is None:
            return _UNKNOWN
        file, target = entry
        if key in self.active or len(self.active) >= self.max_depth or self.calls >= self.max_contexts:
            analyzer.gap('argument_flow_limit', '跨函数参数传播达到深度、上下文或循环边界；不证明该调用后的结果形态。', node)
            return _UNKNOWN
        params = [*target.args.posonlyargs, *target.args.args]
        allowed_keywords = {a.arg for a in [*target.args.args, *target.args.kwonlyargs]}
        if (target.args.vararg or target.args.kwarg or any(isinstance(a, ast.Starred) for a in node.args)
                or len(positional) > len(params) or any(k.arg not in allowed_keywords for k in node.keywords)):
            analyzer.gap('argument_binding_unresolved', '跨函数实参包含动态展开或不支持的签名，需人工确认参数对应关系。', node)
            return _UNKNOWN
        bound = {p.arg: v for p, v in zip(params, positional)}
        for name, value in keywords:
            if name in bound:
                analyzer.gap('argument_binding_unresolved', '参数重复绑定，不能证明有效调用。', node)
                return _UNKNOWN
            bound[name] = value
        required = params[:len(params)-len(target.args.defaults)]
        required += [p for p, d in zip(target.args.kwonlyargs, target.args.kw_defaults) if d is None]
        if any(p.arg not in bound for p in required):
            analyzer.gap('argument_binding_unresolved', '缺少必需参数，不能证明有效调用。', node)
            return _UNKNOWN
        # A shallow cached summary must not bypass a deeper caller's depth/cycle budget.
        identity = (key, tuple(sorted(bound.items())), tuple(self.active))
        if identity in self.cache:
            return self.cache[identity]
        self.calls += 1
        self.active.append(key)
        self.visited_paths.add(file['path'])
        try:
            child = _Analyzer(file, self.evidence, self.report, summaries=self.summaries,
                              flow=self, bound_parameters={id(target): bound})
            tree = self.trees[key[0]]
            child.prepare(tree.body)
            env = {}
            child.block([s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], env)
            # Functions are visible throughout a module once initialization has completed.
            for name, value in child.local_functions.items():
                env.setdefault(name, value)
            child.block([target], env)
            values = child.return_values
            # Conditional returns are not a proof of a single result shape.
            complex_flow = any(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)) for n in ast.walk(target))
            result = values[0] if len(values) == 1 and not complex_flow else _UNKNOWN
            self.cache[identity] = result
            return result
        finally:
            self.active.pop()
