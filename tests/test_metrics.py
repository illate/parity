import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix as sk_cm

from illate_parity.metrics import encode, evaluate


def _random(n=500, k=7, seed=0):
    rng = np.random.default_rng(seed)
    labels = [f"c{i}" for i in range(k)]
    gold = [labels[i] for i in rng.integers(0, k, n)]
    pred = [g if rng.random() < 0.7 else labels[rng.integers(0, k)] for g in gold]
    return gold, pred, labels


def test_matches_sklearn():
    gold, pred, labels = _random()
    ev = evaluate(gold, pred, labels)
    assert ev["accuracy"] == pytest.approx(accuracy_score(gold, pred))
    assert ev["macro_f1"] == pytest.approx(f1_score(gold, pred, average="macro", labels=labels))
    np.testing.assert_array_equal(ev["confusion"][:, :-1], sk_cm(gold, pred, labels=labels))


def test_invalid_prediction_counts_as_wrong():
    ev = evaluate(["a", "b", "a"], ["a", "I am not sure", "a"], ["a", "b"])
    assert ev["accuracy"] == pytest.approx(2 / 3)
    assert ev["invalid"] == 1
    assert ev["top_confusions"][0] == ("b", "<invalid>", 1)


def test_length_mismatch():
    with pytest.raises(ValueError):
        encode(["a"], ["a", "b"])
