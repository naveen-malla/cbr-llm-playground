from cbrlab.similarity.local import sim_numeric, sim_bool, sim_cat

def test_numeric():
    assert sim_numeric(5,5,(0,10)) == 1.0
    assert sim_numeric(0,10,(0,10)) == 0.0

def test_bool():
    assert sim_bool(True, True) == 1.0
    assert sim_bool(True, False) == 0.0

def test_cat():
    assert sim_cat("a","a") == 1.0
    assert sim_cat("a","b") == 0.0
