"""Bounded summaries of straight-line local OCR wrappers; never imports code."""
import ast
from pathlib import PurePosixPath


def local_module(path):
    return str(PurePosixPath(path).with_suffix('')).replace('/', '.')


def import_target(path, node):
    if not node.level:
        return node.module or ''
    parts = local_module(path).split('.')[:-1]
    if node.level > len(parts) + 1:
        return None
    return '.'.join(parts[:len(parts) - node.level + 1] + ([node.module] if node.module else []))


def function_bindings(file, summaries):
    """Keys distinguish import sites, avoiding collisions with unrelated imports."""
    from src.paddleocr_compatibility import _tree
    try:
        tree = _tree(file['content'])
    except SyntaxError:
        return {}, {}
    imported = {}
    own = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = import_target(file['path'], node)
            for alias in node.names:
                value = summaries.get((module, alias.name))
                if value:
                    imported[(node.level, node.module, alias.name)] = ('function_result', value)
        elif isinstance(node, ast.FunctionDef):
            value = summaries.get((local_module(file['path']), node.name))
            if value:
                own[node.name] = ('function_result', value)
    return imported, own


def summarize_functions(files, evidence):
    from src.paddleocr_compatibility import _tree, _Analyzer
    summaries = {}
    candidates = []
    forbidden = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match,
                 ast.AsyncFunctionDef, ast.Lambda, ast.Yield, ast.YieldFrom,
                 ast.Global, ast.Nonlocal, ast.ClassDef)
    for file in files:
        if not file['path'].endswith('.py'):
            continue
        try:
            tree = _tree(file['content'])
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                continue
            if sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name==node.name for n in tree.body)!=1:
                continue
            outside=[n for statement in tree.body if not isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for n in ast.walk(statement)]
            if any(isinstance(n,ast.Name) and n.id==node.name and isinstance(n.ctx,(ast.Store,ast.Del)) for n in outside):
                continue
            children = list(ast.walk(node))
            if any(isinstance(n, forbidden) for n in children) or sum(isinstance(n, ast.FunctionDef) for n in children) != 1:
                continue
            # One terminal return; early exits are not a straight-line proof.
            if not node.body or not isinstance(node.body[-1], ast.Return) or sum(isinstance(n, ast.Return) for n in children) != 1:
                continue
            if len(candidates) < 200:
                candidates.append((file, tree, node))
    # At most six wrapper layers, with no cyclic fixed-point assumptions.
    for _ in range(6):
        changed = False
        for file, tree, node in candidates:
            key = (local_module(file['path']), node.name)
            if key in summaries:
                continue
            scratch = {'findings': [], 'gaps': []}
            analyzer = _Analyzer(file, evidence, scratch, summaries=summaries)
            analyzer.prepare(tree.body)
            env = {}
            analyzer.block([n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], env)
            analyzer.block([node], env)
            if not scratch['gaps'] and analyzer.return_values in ([('ocr_results',)], [('structure_results',)]):
                summaries[key] = analyzer.return_values[0]
                changed = True
        if not changed:
            break
    return summaries
