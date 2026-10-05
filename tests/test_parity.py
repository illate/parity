import pytest

from illate_parity.parity import check, verdict


def _rows(labels, prefix="i"):
    return [{"id": f"{prefix}{k}", "label": lab} for k, lab in enumerate(labels)]


def test_verdict_rules():
    assert verdict(-0.01, 0.02, 0.03) == "PASS"
    assert verdict(-0.10, -0.05, 0.03) == "FAIL"
    assert verdict(-0.05, 0.01, 0.03) == "INCONCLUSIVE"


def test_identical_models_pass():
    gold = [{"id": f"i{k}", "text": "x", "label": "ab"[k % 2]} for k in range(1200)]
    preds = _rows(["ab"[k % 2] if k % 10 else "ba"[k % 2] for k in range(1200)])
    r = check(gold, preds, preds, n_boot=500)
    assert r["verdict"] == "PASS"
    assert r["diff"]["diff"] == 0
    assert r["warnings"] == []


def test_much_worse_candidate_fails():
    gold = [{"id": f"i{k}", "text": "x", "label": "ab"[k % 2]} for k in range(1200)]
    good = _rows(["ab"[k % 2] for k in range(1200)])
    bad = _rows(["ab"[k % 2] if k % 4 else "ba"[k % 2] for k in range(1200)])
    assert check(gold, good, bad, n_boot=500)["verdict"] == "FAIL"


def test_missing_prediction_is_an_error():
    gold = [{"id": "i0", "text": "x", "label": "a"}, {"id": "i1", "text": "y", "label": "b"}]
    with pytest.raises(ValueError, match="no prediction"):
        check(gold, _rows(["a", "b"]), _rows(["a"]), n_boot=100)


def test_agreement_mode_and_small_sample_warning():
    a = _rows(["a", "b"] * 50)
    r = check(None, a, a, n_boot=200)
    assert r["mode"] == "agreement"
    assert r["agreement"]["value"] == 1.0
    assert any("Only 100 items" in w for w in r["warnings"])
