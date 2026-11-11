"""Gemini (via OpenRouter) zero-shot intent classifier."""
from __future__ import annotations

import time

from h5.prompts import build_prompt, parse_label

MODEL = "google/gemini-3.5-flash-lite"
MAX_ATTEMPTS = 3
# NOTE: OpenRouter returns 400 "Reasoning is mandatory for this endpoint and cannot be
# disabled" for reasoning.enabled=False on this model, so the field is omitted.


def classify(utterance: str, labels: list[str], definitions: dict[str, str] | None, client) -> str | None:
    prompt = build_prompt(utterance, labels, definitions)
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=32,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return parse_label(content, labels)
        except Exception:
            pass
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(2 ** attempt)
    return None
