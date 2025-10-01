from cbrlab.retrieval import Embedder

def test_embedder_fallback():
    e = Embedder()
    e.fit(["apple banana", "table chair"])
    idxs, _ = e.topk("apple", k=1)
    assert isinstance(idxs, list) and len(idxs) == 1
