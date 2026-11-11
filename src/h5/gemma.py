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


def build_label_trie(label_token_seqs: dict[str, list[int]]) -> dict:
    """Prefix trie over label token ids. Node = {"next": {token: node}, "label": str | None}."""
    root: dict = {"next": {}, "label": None}
    for label, seq in label_token_seqs.items():
        node = root
        for t in seq:
            node = node["next"].setdefault(t, {"next": {}, "label": None})
        node["label"] = label
    return root


def predict_constrained(model, tokenizer, utterances: list[str], labels: list[str]) -> list[tuple[str, float]]:
    """Greedy decoding restricted (via a token trie) to the given labels, so the output is always valid.

    Returns (label, logprob) per utterance; logprob is the unconstrained model log-probability of the
    chosen label tokens followed by <end_of_turn>, usable as a confidence score.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    eot = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    assert len(eot) == 1, eot
    eot = eot[0]
    trie = build_label_trie({l: tokenizer.encode(l, add_special_tokens=False) for l in labels})
    out = []
    for u in utterances:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(u, labels)}],
            add_generation_prompt=True, tokenize=False,
        )
        ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))[None]
        cache = make_prompt_cache(model)
        node, logp, label = trie, 0.0, None
        logits = model(ids, cache=cache)[:, -1, :]
        while True:
            lp = logits[0] - mx.logsumexp(logits[0])
            allowed = list(node["next"])
            if node["label"] is not None:
                allowed.append(eot)
            mask = mx.full(lp.shape, -mx.inf)
            mask[mx.array(allowed)] = 0
            tok = int(mx.argmax(lp + mask).item())
            logp += float(lp[tok].item())
            if tok == eot:
                label = node["label"]
                break
            node = node["next"][tok]
            if not node["next"]:  # unique completion: score the end token too, then stop
                logits = model(mx.array([[tok]]), cache=cache)[:, -1, :]
                lp = logits[0] - mx.logsumexp(logits[0])
                logp += float(lp[eot].item())
                label = node["label"]
                break
            logits = model(mx.array([[tok]]), cache=cache)[:, -1, :]
        out.append((label, logp))
    return out
