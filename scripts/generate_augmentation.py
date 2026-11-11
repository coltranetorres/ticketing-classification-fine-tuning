"""Generate diverse synthetic training messages with Gemini (OpenRouter), then keep only those
Gemini itself classifies back to the intended intent (self-consistency filter).

Prompts contain ONLY label names, label definitions (h5.prompts) and style instructions: no holdout
or Bitext text, so test/dev wording cannot leak into training data.

Usage: uv run python scripts/generate_augmentation.py [--intents a,b] [--batches N] [--per-batch N]
Writes data/aug/raw.jsonl (all generated) and data/aug/augmented.jsonl (filtered, deduped).
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from h5.data import LABELS, load_holdout, load_split
from h5.gemini import classify
from h5.prompts import LABEL_DEFINITIONS

MODEL = "google/gemini-3.5-flash-lite"
OUT = Path("data/aug")

STYLES = [
    "very short and casual: lowercase, no punctuation, typos, text-speak, 3 to 10 words",
    "long and detailed: 2 to 4 sentences with backstory and concrete details such as product names, dates, order numbers",
    "polite and formal, like an email from a customer who writes carefully",
    "indirect and implicit: describe the situation or the goal WITHOUT using the obvious keywords from the intent name",
    "frustrated or anxious tone, emotional, maybe ALL CAPS or exclamation marks in places",
    "non-native English speaker, slightly unusual grammar or word choice, or heavy abbreviations",
]

# intents most often mistaken for each other; shown to the generator so messages stay unambiguous
NEIGHBORS = {
    "get_invoice": ["check_invoice"], "check_invoice": ["get_invoice"],
    "contact_customer_service": ["contact_human_agent"], "contact_human_agent": ["contact_customer_service"],
    "cancel_order": ["check_cancellation_fee", "change_order"], "check_cancellation_fee": ["cancel_order"],
    "get_refund": ["track_refund", "check_refund_policy"], "track_refund": ["get_refund"],
    "check_refund_policy": ["get_refund"], "set_up_shipping_address": ["change_shipping_address"],
    "change_shipping_address": ["set_up_shipping_address"], "create_account": ["registration_problems"],
    "registration_problems": ["create_account"], "edit_account": ["switch_account"],
    "switch_account": ["edit_account"], "change_order": ["cancel_order"],
    "delivery_options": ["delivery_period"], "delivery_period": ["delivery_options", "track_order"],
    "track_order": ["delivery_period"], "payment_issue": ["check_payment_methods", "complaint"],
    "check_payment_methods": ["payment_issue"], "complaint": ["review"], "review": ["complaint"],
}


def gen_prompt(intent: str, style: str, n: int) -> str:
    neigh = "".join(f"\n- {o}: {LABEL_DEFINITIONS[o]}" for o in NEIGHBORS.get(intent, []))
    return (
        f"You write realistic customer messages sent to an online shop's support channel.\n"
        f"Write {n} DIFFERENT messages whose intent is exactly: {intent}\n"
        f"Meaning of {intent}: {LABEL_DEFINITIONS[intent]}\n"
        + (f"Do NOT write messages that fit these similar intents instead:{neigh}\n" if neigh else "")
        + f"Style for all {n} messages: {style}.\n"
        f"Vary topics, products and wording; no two messages should be near-duplicates. "
        f"Each message must be a complete customer message that clearly expresses {intent}.\n"
        f"Return ONLY a JSON array of {n} strings."
    )


def parse_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in arr if isinstance(s, str) and 3 <= len(s.strip()) <= 400]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents", default=",".join(LABELS))
    ap.add_argument("--batches", type=int, default=len(STYLES))
    ap.add_argument("--per-batch", type=int, default=20)
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    intents = args.intents.split(",")
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(i, s) for i in intents for s in STYLES[: args.batches]]

    def gen(job):
        intent, style = job
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=1.0, max_tokens=6000,
                    messages=[{"role": "user", "content": gen_prompt(intent, style, args.per_batch)}],
                )
                msgs = parse_list(r.choices[0].message.content)
                if msgs:
                    return [{"utterance": m, "intent": intent, "style": style} for m in msgs]
            except Exception:
                pass
        return []

    with ThreadPoolExecutor(8) as ex:
        raw = [row for rows in ex.map(gen, jobs) for row in rows]
    print(f"generated {len(raw)} raw messages from {len(jobs)} calls")
    with open(OUT / "raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    # drop exact duplicates and anything equal to Bitext / holdout text (case-insensitive)
    banned = {r["utterance"].lower().strip() for s in ("train", "val", "test") for r in load_split(s)}
    banned |= {r["utterance"].lower().strip() for v in (1, 2) for r in load_holdout(v)}
    seen, cand = set(), []
    for r in raw:
        k = r["utterance"].lower().strip()
        if k in banned or k in seen:
            continue
        seen.add(k)
        cand.append(r)
    print(f"after dedupe/overlap removal: {len(cand)}")

    if args.no_filter:
        kept = cand
    else:
        lock = threading.Lock()
        stats = {"n": 0}

        def check(r):
            pred = classify(r["utterance"], LABELS, LABEL_DEFINITIONS, client)
            with lock:
                stats["n"] += 1
            return pred == r["intent"]

        with ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(check, cand))
        kept = [r for r, good in zip(cand, ok) if good]
        print(f"kept after Gemini self-consistency filter: {len(kept)} of {len(cand)}")
    with open(OUT / "augmented.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["intent"] for r in kept)
    print("per-intent kept (min/max):", min(c.values()) if c else 0, max(c.values()) if c else 0)


if __name__ == "__main__":
    main()
