"""Render selected original windows, never ranking context or an unrelated tail."""

def display_spans(row:dict)->list[dict]:
    text=row.get('content','');spans=row.get('evidence_spans')
    if not spans:
        return [{'content':text,'line_start':row.get('line_start'),'line_end':row.get('line_end')}]
    base=row.get('line_start');lines=text.splitlines();out=[]
    if type(base) is not int or not isinstance(spans,list):raise ValueError('invalid source range')
    for span in sorted(spans,key=lambda s:s.get('char_start',s.get('line_start',0))):
        start,end=span.get('line_start'),span.get('line_end')
        if type(start) is not int or type(end) is not int or not base<=start<=end<base+len(lines):
            raise ValueError('span outside original source lines')
        if 'char_start' in span or 'char_end' in span:
            lo,hi=span.get('char_start'),span.get('char_end')
            if type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=len(text):
                raise ValueError('span outside original source characters')
            if base+text[:lo].count('\n')!=start or base+text[:hi-1].count('\n')!=end:
                raise ValueError('span character/line mismatch')
            value=text[lo:hi]
        else:value='\n'.join(lines[start-base:end-base+1])
        if span.get('content',value)!=value:raise ValueError('span text mismatch')
        out.append({'content':value,'line_start':start,'line_end':end})
    return out
