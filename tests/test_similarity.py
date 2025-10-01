from cbrlab.similarity.global_similarity import aggregate
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


def test_global_similarity_respects_query_features():
    schema = {
        "features": {
            "voltage_vdd": {"type": "numeric", "range": [0, 30]},
            "relay_switched": {"type": "boolean"},
        },
        "global_weights": {"voltage_vdd": 0.6, "relay_switched": 0.4},
    }
    candidate = {"voltage_vdd": 24.0, "relay_switched": True}
    same_query = {"voltage_vdd": 24.0, "relay_switched": True}
    diff_query = {"voltage_vdd": 18.0, "relay_switched": False}

    same_score, same_per = aggregate(same_query, candidate, schema)
    diff_score, diff_per = aggregate(diff_query, candidate, schema)

    assert same_per["voltage_vdd"] == 1.0
    assert diff_per["voltage_vdd"] < same_per["voltage_vdd"]
    assert diff_score < same_score
