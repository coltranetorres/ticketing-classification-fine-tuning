import pytest

from h5.data import LABELS
from h5.prompts import LABEL_DEFINITIONS, build_prompt, parse_label

L = LABELS


@pytest.mark.parametrize(
    "text,expected",
    [
        ("get_invoice", "get_invoice"),
        ("Intent: **Get_Invoice**.", "get_invoice"),
        ("`track_refund`\n", "track_refund"),
        ("get_refund or track_refund", None),
        ("check_invoice", "check_invoice"),
        ("", None),
        (None, None),
        ("no idea", None),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text, L) == expected


def test_definitions_cover_labels():
    assert set(LABEL_DEFINITIONS) == set(LABELS)


def test_build_prompt_contents():
    p = build_prompt("where is my stuff", L)
    assert "where is my stuff" in p
    assert all(l in p for l in L)
    assert "Answer with exactly one label." in p
    assert LABEL_DEFINITIONS["get_invoice"] not in p


def test_build_prompt_with_definitions():
    p = build_prompt("hi", L, LABEL_DEFINITIONS)
    assert all(f"{l}: {LABEL_DEFINITIONS[l]}" in p for l in L)
