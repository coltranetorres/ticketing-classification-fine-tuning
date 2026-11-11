import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).resolve().parents[1] / "scripts" / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_mcnemar_hand_computed():
    assert abs(report.mcnemar_p(5, 1) - 0.21875) < 1e-12
    assert report.mcnemar_p(1, 5) == report.mcnemar_p(5, 1)


def test_mcnemar_no_discordant_and_extreme():
    assert report.mcnemar_p(0, 0) == 1.0
    assert abs(report.mcnemar_p(10, 0) - 2 / 1024) < 1e-12
    assert report.mcnemar_p(3, 3) == 1.0


def test_confusion_counts_excludes_correct_and_names_none():
    rows = [
        {"true": "a", "pred": "b"}, {"true": "a", "pred": "b"},
        {"true": "a", "pred": "a"}, {"true": "c", "pred": None},
        {"true": "c", "pred": "a"},
    ]
    top = report.top_confusions(rows, k=10)
    assert top[0] == ("a", "b", 2)
    assert ("c", "(unparseable)", 1) in top
    assert all(t != p for t, p, _ in top)
    assert len(report.top_confusions(rows, k=1)) == 1


def test_paired_counts():
    t = [{"true": "a", "pred": "a"}, {"true": "a", "pred": "b"}, {"true": "a", "pred": "a"}]
    g = [{"true": "a", "pred": "b"}, {"true": "a", "pred": "a"}, {"true": "a", "pred": "a"}]
    assert report.paired_counts(t, g) == (1, 1)


def test_verdict():
    m = lambda a: {"accuracy": a}
    assert report.verdict(m(0.9), m(0.8), m(0.85)).startswith("H5 supported")
    assert report.verdict(m(0.9), m(0.9), m(0.8)).startswith("H5 not supported")
    assert report.verdict(m(0.9), m(0.95), m(0.8)).startswith("H5 not supported")
