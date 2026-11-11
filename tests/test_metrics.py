from h5.metrics import CONFUSABLE_PAIRS, score

LABELS = ["a", "b", "get_invoice", "check_invoice", "cancel_order"]


def test_perfect():
    y = list(LABELS)
    r = score(y, list(y), LABELS)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0 and r["n"] == 5
    assert r["unparseable"] == 0


def test_none_is_wrong_and_counted():
    r = score(["a", "b"], ["a", None], LABELS)
    assert r["accuracy"] == 0.5
    assert r["unparseable"] == 1
    # None must not add a class: macro over 5 labels, a F1=1, b F1=0, rest 0
    assert abs(r["macro_f1"] - 1 / 5) < 1e-9


def test_confusable_accuracy_ignores_other_rows():
    yt = ["get_invoice", "check_invoice", "a", "b"]
    yp = ["get_invoice", "get_invoice", "a", "a"]
    assert score(yt, yp, LABELS)["confusable_accuracy"] == 0.5


def test_no_confusable_rows_returns_zero():
    assert score(["a"], ["a"], LABELS)["confusable_accuracy"] == 0.0


def test_empty():
    r = score([], [], LABELS)
    assert r["n"] == 0 and r["accuracy"] == 0.0 and r["macro_f1"] == 0.0


def test_pairs():
    assert len(CONFUSABLE_PAIRS) == 4
