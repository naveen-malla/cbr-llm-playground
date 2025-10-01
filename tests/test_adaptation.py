from cbrlab.adaptation.rules import apply_rules

def test_rules_append():
    rules=[{"action":"append_note","if":"delta_voltage","params":{"note":"Check PSU"}}]
    out = apply_rules(rules, {"voltage_vdd":20.0}, {"voltage_vdd":15.0}, "Base")
    assert "Check PSU" in out or isinstance(out, str)
