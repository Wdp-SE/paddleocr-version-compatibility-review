"""Presentation only: retain factual text and distinguish source identities."""
import re
from html import unescape


def evidence_excerpt(texts, query='', limit=180):
    """Display an extract of a verified span, without HTML table markup."""
    terms = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]{2,}', query.casefold()))
    candidates = []
    for text in texts:
        plain = re.sub(r'</?(?:table|tr|td|th|thead|tbody|ul|ol|li|p|div|span|code|pre|b|strong|em|i|br)\b[^>]*>', ' ', text, flags=re.I)
        plain = re.sub(r'\s+', ' ', unescape(plain)).strip()
        matches = [plain.casefold().find(term) for term in terms if term in plain.casefold()]
        candidates.append((len(matches), plain, min(matches) if matches else 0))
    if not candidates:
        return ''
    _, plain, position = max(candidates, key=lambda item: item[0])
    start = max(0, position - 45)
    return ('…' if start else '') + plain[start:start + limit] + ('…' if start + limit < len(plain) else '')


def paragraph_blocks(text, target_chars=120):
    """Break at prose sentence ends, never inside inline or fenced code."""
    blocks, pending = [], ''
    for part in re.split(r'(```[\s\S]*?```|`[^`\n]*`)', text):
        pieces = [part] if part.startswith('`') else re.split(r'(?<=[。！？；])', part)
        for piece in pieces:
            pending += piece
            if len(pending) >= target_chars and not piece.startswith('`') and piece.endswith(('。', '！', '？', '；')):
                blocks.append(pending)
                pending = ''
    if pending:
        blocks.append(pending)
    return blocks


def unique_citations(citations):
    """Merge exact provenance/location duplicates; incomplete records stay separate."""
    fields = ('version', 'path', 'url', 'commit', 'sha256', 'line_start', 'line_end')
    seen, result = set(), []
    for citation in citations:
        key = tuple(citation.get(field) for field in fields)
        complete = all(value is not None and value != '' for value in key)
        if not complete or key not in seen:
            result.append(citation)
        if complete:
            seen.add(key)
    return result
