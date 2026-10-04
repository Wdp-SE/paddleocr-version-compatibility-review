"""Source-span retrieval metrics, explicitly separate from answer accuracy."""
def assess_retrieval(case, hits):
    matches=[]
    for fact in case['facts']:
        matches.append([i+1 for i,row in enumerate(hits)
                        if row['version']==case['version'] and row['document_path']==fact['path']
                        and all(token in row['content'] for token in fact['tokens'])])
    found=sum(bool(rows) for rows in matches)
    ranks=[r for rows in matches for r in rows]
    return {'fact_recall':found/len(matches),'complete':found==len(matches),
            'reciprocal_rank':1/min(ranks) if ranks else 0,
            'matched_fact_count':found,'required_fact_count':len(matches),
            'wrong_version_count':sum(row['version']!=case['version'] for row in hits)}


def select_strategy(results):
    eligible=[policy for policy,values in results.items() if values['dev']['wrong_version_count']==0]
    if not eligible:raise ValueError('no version-safe strategy')
    # Completeness first, then recall. Stable insertion order is the simplicity tie-break.
    return max(eligible,key=lambda p:(results[p]['dev']['complete_rate'],results[p]['dev']['fact_recall']))
