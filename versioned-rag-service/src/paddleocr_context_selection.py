"""Prefer actual symbol definitions over repeated heading context within selected sources."""
import re
import ast
import hashlib
import textwrap


def complete_parameter_context(views, parents, *, max_chars=1536):
    """Complete a literal or short function in the same retrieved parent.

    This is bounded source-context expansion, never an inferred acceptance
    verdict. Partial, unparsable or oversized assignments stay unchanged.
    """
    from src.paddleocr_retrieval_views import _readable
    result = []
    for view in views:
        parent = parents[view['parent_chunk_id']]
        text = parent['content']
        replacement = view
        if parent.get('document_path', '').endswith('.py'):
            try:
                tree = ast.parse(textwrap.dedent(text))
            except (SyntaxError, ValueError):
                tree = None
            if tree is not None:
                lines = text.splitlines(keepends=True)
                offsets = [0]
                for line in lines:
                    offsets.append(offsets[-1] + len(line))
                for node in ast.walk(tree):
                    literal = isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict)
                    # A short routing loop/conditional is one proof unit. A
                    # window containing its first branch cannot prove how the
                    # other branch forwards or rejects an unknown keyword.
                    function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.If))
                    if not (literal or function):
                        continue
                    lo, hi = offsets[node.lineno-1], offsets[node.end_lineno]
                    # A literal must start in the window; a short function must
                    # contain it. Do not fetch unrelated code from the file.
                    contains = (lo <= view['char_start'] < hi if function else
                                view['char_start'] <= lo < view['char_end'])
                    # Preserve the whole original window when completing it.
                    lo, hi = min(lo, view['char_start']), max(hi, view['char_end'])
                    if not (contains and hi-lo <= max_chars):
                        continue
                    value = text[lo:hi]
                    # Context may itself contain newlines: retain exactly its
                    # recorded length rather than reconstructing its heading.
                    prefix = view['retrieval_text'][:view.get('context_char_end', 0)]
                    replacement = {**view, 'char_start': lo, 'char_end': hi, 'content': value,
                        'line_start': parent['line_start']+text[:lo].count('\n'),
                        'line_end': parent['line_start']+text[:hi-1].count('\n'),
                        'view_id': hashlib.sha256(f"{parent['chunk_id']}:{lo}:{hi}".encode()).hexdigest(),
                        'retrieval_text': (prefix+'\n' if prefix else '') +
                            (_readable(value) if view.get('representation') == 'structured' else value)}
                    replacement['token_count'] = len(replacement['retrieval_text'])
                    break
        result.append(replacement)
    return result


def parameter_contract_windows(views):
    """Keep retrieved keyword routing, accepted keys and rejection in one pack.

    These are unmodified source windows, not an inferred parameter verdict.
    No source is loaded outside the ranked, version-filtered candidate pool.
    """
    anchors = (
        ('_DEPRECATED_PARAM_NAME_MAPPING =',),
        ('for name, val in kwargs.items()',),
        ('base_params[name] = val',),
        ('super().__init__(**base_params)',),
        ('self._common_args = parse_common_args(',),
        ('default_vals =',),
        ('unknown_names', 'Unknown argument'),
    )
    result = []
    for anchor in anchors:
        for version in dict.fromkeys(v['version'] for v in views):
            found = next((v for v in views if v['version'] == version and
                          all(token in v['content'] for token in anchor)), None)
            if found and found not in result:
                result.append(found)
    return result


def initializer_context(views,parents):
    """Copy a bounded constructor signature from an already retrieved parent."""
    result=[];seen=set()
    for view in views:
        key=view['parent_chunk_id']
        if key in seen:continue
        seen.add(key);parent=parents[key];text=parent['content']
        if not parent.get('document_path','').endswith('.py'):continue
        match=re.search(r'(?m)^[ \t]*def __init__\(',text)
        if not match:continue
        end=text.find('):',match.end())
        if end<0 or end+2-match.start()>1536:continue
        lo,hi=match.start(),end+2;value=text[lo:hi]
        result.append({**view,'content':value,'char_start':lo,'char_end':hi,
            'line_start':parent['line_start']+text[:lo].count('\n'),
            'line_end':parent['line_start']+text[:hi-1].count('\n'),
            'view_id':hashlib.sha256(f'{key}:{lo}:{hi}'.encode()).hexdigest(),
            'retrieval_text':value,'contract_anchor':True,'context_priority':3})
    return result


def definition_score(view,symbols):
    text=view.get('content','')
    present=[s for s in symbols if s!='PaddleOCR' and re.search(r'(?<![\w])'+re.escape(s)+r'(?![\w])',text)]
    definition=bool(present and (re.search(r'<td\b|\|.*\|',text,re.I)
        or any(re.search(re.escape(s)+r'(?:`|</code>)?\s*[:=]',text) for s in present)
        or 'add_argument(' in text and 'default=' in text))
    return int(definition),len(present)


def select_context(views,*,symbols,top_k,max_windows):
    parents=list(dict.fromkeys(v['parent_chunk_id'] for v in views))[:top_k]
    symbols=list(dict.fromkeys(s for s in symbols if s!='PaddleOCR'))
    chosen=[]
    for parent in parents:
        candidates=[v for v in views if v['parent_chunk_id']==parent]
        def score(view):
            return definition_score(view,symbols)
        # No evidence is added or rewritten. Parent order preserves requirement reservations.
        candidates=sorted(enumerate(candidates),key=lambda pair:(
            -pair[1].get('context_priority',0),-int(pair[1].get('contract_anchor', False)), *[-n for n in score(pair[1])], pair[0]))
        chosen.extend(v for _,v in candidates[:max_windows])
    return chosen
