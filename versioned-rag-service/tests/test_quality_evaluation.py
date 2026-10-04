def test_metrics_require_all_facts_and_penalize_wrong_version():
    from src.quality_evaluation import assess_retrieval
    case={'version':'v3','facts':[{'path':'a','tokens':['one']},{'path':'b','tokens':['two']}]}
    result=assess_retrieval(case,[{'document_path':'a','version':'v3','content':'one'},
                                  {'document_path':'b','version':'v2','content':'two'}])
    assert result['fact_recall']==.5
    assert result['complete'] is False
    assert result['wrong_version_count']==1


def test_selection_is_made_on_dev_not_holdout_or_speed_alone():
    from src.quality_evaluation import select_strategy
    results={'bm25':{'dev':{'fact_recall':.6,'complete_rate':.5,'wrong_version_count':0}},
             'hybrid_rerank':{'dev':{'fact_recall':.8,'complete_rate':.7,'wrong_version_count':0}},
             'bad':{'dev':{'fact_recall':1.,'complete_rate':1.,'wrong_version_count':1}}}
    assert select_strategy(results)=='hybrid_rerank'
