from cbrlab.similarity import aggregate, sim_bool, sim_cat, sim_numeric

def test_numeric():
    assert sim_numeric(5,5,(0,10)) == 1.0
    assert sim_numeric(0,10,(0,10)) == 0.0

def test_bool():
    assert sim_bool(True, True) == 1.0
    assert sim_bool(True, False) == 0.0

def test_cat():
    assert sim_cat("a","a") == 1.0
    assert sim_cat("a","b") == 0.0


def test_aggregate_uses_query_features():
    schema = {
        "features": {
            "temperature": {"type": "numeric", "range": [0, 100]},
            "alarm": {"type": "boolean"},
        },
        "global_weights": {"temperature": 0.5, "alarm": 0.5},
    }
    case_problem = {"temperature": 70, "alarm": True}
    matching_score, matching_per = aggregate(case_problem, case_problem, schema)
    assert matching_score == 1.0
    assert all(v == 1.0 for v in matching_per.values())

    query_problem = {"temperature": 10, "alarm": False}
    score, per = aggregate(query_problem, case_problem, schema)
    assert score < matching_score
    assert per["temperature"] < 1.0
    assert per["alarm"] == 0.0
