from collections import Counter

from h5.data import LABELS, load_split, stratified_sample


def test_split_sizes():
    assert len(load_split("train")) == 6539
    assert len(load_split("val")) == 818
    assert len(load_split("test")) == 818


def test_labels():
    assert len(LABELS) == 27
    assert LABELS == sorted(LABELS)


def test_stratified_sample_balanced_and_deterministic():
    train = load_split("train")
    s1 = stratified_sample(train, 1000)
    s2 = stratified_sample(train, 1000)
    assert len(s1) == 1000
    counts = Counter(r["intent"] for r in s1)
    assert set(counts) == set(LABELS)
    assert all(36 <= c <= 38 for c in counts.values())
    assert s1 == s2


def test_small_label_returns_all():
    rows = [{"utterance": f"a{i}", "intent": "a"} for i in range(3)]
    rows += [{"utterance": f"b{i}", "intent": "b"} for i in range(50)]
    out = stratified_sample(rows, 20)
    assert len(out) == 20
    assert sum(r["intent"] == "a" for r in out) == 3


def test_sample_comes_from_train_only():
    train = load_split("train")
    sample = stratified_sample(train, 1000)
    train_pairs = {(r["utterance"], r["intent"]) for r in train}
    assert {(r["utterance"], r["intent"]) for r in sample} <= train_pairs
