from types import SimpleNamespace

from h5 import gemini
from h5.data import LABELS


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=o))])


def test_parses_label(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    assert gemini.classify("x", LABELS, None, FakeClient(["get_invoice"])) == "get_invoice"


def test_none_after_three_failures(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    c = FakeClient([RuntimeError("boom")])
    assert gemini.classify("x", LABELS, None, c) is None
    assert len(c.calls) == 3


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    c = FakeClient([RuntimeError("boom"), "track_order"])
    assert gemini.classify("x", LABELS, None, c) == "track_order"
    assert len(c.calls) == 2


def test_request_kwargs(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    c = FakeClient(["get_invoice"])
    gemini.classify("x", LABELS, None, c)
    kw = c.calls[0]
    assert kw["model"] == "google/gemini-3.5-flash-lite"
    assert "extra_body" not in kw  # reasoning.enabled=False is rejected (400) for this model
    assert kw["temperature"] == 0
