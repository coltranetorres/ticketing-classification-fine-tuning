"""Gemma (MLX) helpers: chat-format training examples and batch-free greedy prediction."""
from __future__ import annotations

from h5.data import Row
from h5.prompts import build_prompt, parse_label


def to_chat_example(row: Row, labels: list[str]) -> dict:
    return {"messages": [
        {"role": "user", "content": build_prompt(row["utterance"], labels)},
        {"role": "assistant", "content": row["intent"]},
    ]}


def predict(model, tokenizer, utterances: list[str], labels: list[str], raw: list | None = None) -> list[str | None]:
    from mlx_lm import generate
    out = []
    for u in utterances:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(u, labels)}],
            add_generation_prompt=True, tokenize=False,
        )
        text = generate(model, tokenizer, prompt=prompt, max_tokens=12, verbose=False)
        # mlx does not always stop on Gemma's <end_of_turn>; keep only the first turn.
        text = text.split("<end_of_turn>")[0].strip()
        if raw is not None:
            raw.append(text)
        out.append(parse_label(text, labels))
    return out
