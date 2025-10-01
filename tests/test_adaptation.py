from app.rules import apply_rules


def test_voltage_mismatch_appends_note():
    source_case = {
        "id": "case-001",
        "name": "12V Panel Relay",
        "voltage_rating": 12,
        "current_rating": 2.0,
        "environment": "indoor",
    }

    query = {
        "voltage_rating": 24,
        "current_rating": 2.0,
        "environment": "indoor",
    }

    adapted_case, notes = apply_rules(source_case, query)

    assert adapted_case["voltage_rating"] == 24
    assert any("voltage" in note.lower() for note in notes), "Expected voltage mismatch note to be added"
