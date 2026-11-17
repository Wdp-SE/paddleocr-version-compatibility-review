"""Ranking representations only; original evidence bytes and identities stay unchanged."""
import re

from src.public_knowledge import tokens as base_tokens
from src.paddleocr_retrieval_views import _readable


def retrieval_tokens(text):
    result=[]
    for token in base_tokens(text):
        result.append(token)
        if re.match(r'^[a-z]',token) and re.search(r'[_.]',token):
            result.extend(dict.fromkeys(t for t in re.split(r'[_.]+',token) if len(t)>=2))
    return result


def is_layout_only(content,path):
    if not path.lower().endswith('.md'):
        return False
    body=content.strip()
    return (not _readable(body)
        or bool(re.fullmatch(r'---\s*\n.*?\n---',body,re.S))
        or bool(body) and all(line.lstrip().startswith('#') for line in body.splitlines() if line.strip())
        or bool(re.fullmatch(r'<tr\b[^>]*>\s*(?:<th\b.*?</th>\s*)+</tr>',body,re.S|re.I)))


def parameter_definition_priority(content,symbols):
    cells=re.findall(r'<td\b[^>]*>(.*?)</td>',content,re.S|re.I)
    name=_readable(cells[0]) if cells else ''
    if not cells and content.lstrip().startswith('|'):
        name=content.strip().strip('|').split('|')[0].strip().strip('`')
    return int(name in symbols and (len(cells)>1 or '|' in content))


def query_definition_symbols(query):
    if re.search(r'源码|实现|调用链|代码位置|签名',query):
        return set()
    return set(re.findall(r'\b[A-Za-z]\w*_\w+\b',query))


def behavior_context_priority(content,symbols):
    """Prefer a literal comparison over a symbol mention, never infer its result."""
    return int(any(re.search(r'\bif\b[^\n]*\b'+re.escape(symbol)+r'\b[^\n]*:',content)
                   and re.search(r'\bif\b[^\n]*(?:>=|<=|==|>|<|\band\b|\bor\b|\bnot\b)[^\n]*:',content)
                   for symbol in symbols))


def task_definition_priority(content,intent):
    """Prefer actual definition rows matching a task's documented vocabulary."""
    if intent.get('kind')!='parameter_lookup':return 0
    cells=re.findall(r'<td\b[^>]*>(.*?)</td>',content,re.S|re.I)
    if not cells and content.lstrip().startswith('|'):
        cells=content.strip().strip('|').split('|')
    if len(cells)<2 or not re.fullmatch(r'[A-Za-z]\w*',_readable(cells[0]).strip('`')):
        return 0
    # Source-discovery aliases may name implementation variables. They are not
    # requirements that every parameter definition row must contain.
    aliases=' '.join(intent.get('definition_aliases') or intent.get('aliases',[]))
    words=set(re.findall(r'[A-Za-z]+',aliases.lower()))
    body=_readable(content)
    stages=[stage for stage in ('识别','检测') if stage in aliases]
    name = _readable(cells[0]).strip('`')
    batch_name = {'batch', 'size'}.issubset(words) and 'batch' in re.split('_', name.lower())
    return int(bool(words) and (words.issubset(retrieval_tokens(body)) or batch_name)
               and all(stage in body for stage in stages))
