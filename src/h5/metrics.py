"""Scoring for intent classification runs."""
from __future__ import annotations

from sklearn.metrics import f1_score

CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("get_invoice", "check_invoice"),
    ("contact_customer_service", "contact_human_agent"),
    ("cancel_order", "check_cancellation_fee"),
    ("get_refund", "track_refund"),
]
_CONFUSABLE = {label for pair in CONFUSABLE_PAIRS for label in pair}
_SENTINEL = "__unparseable__"


def score(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    """Return accuracy, macro_f1, confusable_accuracy, unparseable, n.

    None predictions count as wrong. Macro-F1 is averaged over `labels` only.
    Edge cases: empty input -> 0.0 metrics; if no row has a true label in a
    confusable pair, confusable_accuracy is 0.0 (not None).
    """
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred must have equal length")
    unparseable = sum(p is None for p in y_pred)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "confusable_accuracy": 0.0,
                "unparseable": 0, "n": 0}
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n
    preds = [_SENTINEL if p is None else p for p in y_pred]
    macro_f1 = float(f1_score(y_true, preds, labels=labels, average="macro", zero_division=0))
    conf = [(t, p) for t, p in zip(y_true, y_pred) if t in _CONFUSABLE]
    conf_acc = sum(t == p for t, p in conf) / len(conf) if conf else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1,
            "confusable_accuracy": conf_acc, "unparseable": unparseable, "n": n}
