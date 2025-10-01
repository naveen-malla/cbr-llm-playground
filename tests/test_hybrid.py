from cbrlab.hybrid.rag import fuse
def test_fuse():
    order, fused = fuse([0.1,0.9],[0.9,0.1], alpha=0.5)
    assert order[0] in (0,1)
    assert len(fused)==2
