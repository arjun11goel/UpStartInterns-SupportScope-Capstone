from supportscope.retrieval import PolicyRetriever


def test_refund_query_retrieves_refund_policy():
    results = PolicyRetriever().retrieve("How long does my refund take after inspection?", top_k=2)
    assert results
    assert results[0].source_id == "REFUND-01"


def test_out_of_scope_query_has_no_policy_match():
    results = PolicyRetriever().retrieve("Explain recursion in Python with an example.", top_k=2)
    assert results == []


def test_damaged_item_query_retrieves_damaged_policy():
    results = PolicyRetriever().retrieve("My parcel arrived broken; should I send photos?", top_k=2)
    assert results
    assert results[0].source_id == "DAMAGED-01"
