"""Present verified experiment data, not theoretical or fallback scores."""
import streamlit as st

LABELS={'raw_bm25':'原始 BM25','window_bm25':'窗口 BM25＋多查询 RRF',
    'structure_bm25':'结构 BM25＋多查询 RRF','structure_hybrid':'结构混合检索（BM25＋BGE）',
    'structure_bm25_rerank':'结构 BM25＋BGE 重排','structure_hybrid_rerank':'结构混合检索＋BGE 重排'}


def render_comparison(report):
    st.markdown('**BM25 与混合检索对比**')
    if not isinstance(report,dict):
        st.caption('尚无与当前检索代码、语料指纹一致的对照结果；不展示过期指标。')
        return
    rows=[]
    st.caption('本地离线对照；本次请求实际使用的策略以结果诊断为准。后端未配置神经模型时会明确回退，不能沿用神经成绩描述该请求。')
    for row in report.get('rows',[]):
        if row.get('evaluation_status','complete')!='complete':continue
        count=row['count'];success=row['execution_success_count']
        rows.append({'策略':LABELS.get(row['strategy'],row['strategy']),
            '题集':{'regression':'已知回归','validation':'新增验证','sealed':'封存题复验' if report.get('conditions',{}).get('sealed_status') else '首次封存检验'}.get(row['split'],row['split']),
            '完整证据覆盖':f"{row['complete_count']}/{count}",
            '错版片段':row['wrong_version_count'],
            '实际执行':f'{success}/{count}' if success==count else f'{success}/{count}（有回退，不能视作完整神经成绩）',
            '检索 P50（秒）':round(row.get('latency_p50_ms',0)/1000,3)})
    st.dataframe(rows,hide_index=True,width='stretch')
    st.write('选型：'+LABELS.get(report.get('selected'),str(report.get('selected')))+'。'+report.get('selection_reason',''))
    st.caption(report.get('limitations','证据覆盖不等于回答正确率。'))
    if report.get('rerank_readiness',{}).get('status')=='NOT_QUALIFIED':
        st.warning('神经重排尚未通过准入：'+report['rerank_readiness']['reason'])
    for name,details in report.get('disqualified',{}).items():
        st.warning(f"{LABELS.get(name,name)}未通过可靠性准入：{details['reason']}。已执行 {details['observed']}/{details['planned']} 题后中止；不提供全量质量指标，不纳入默认选型。")
    with st.expander('实验条件与失败诊断'):
        st.json({k:report.get(k) for k in ('conditions','models','failures','generation_evaluation')})
