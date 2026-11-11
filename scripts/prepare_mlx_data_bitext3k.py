"""Ablation: same size as the augmented run (3,160 train rows) but ALL Bitext: the 1,000 stratified rows
plus 2,160 more drawn (stratified-ish, 80/intent) from the rest of Bitext train. Valid = same as mlx_aug."""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from h5.data import LABELS, load_split, stratified_sample
from h5.gemma import to_chat_example

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mlx_bitext3k"


def main():
    train = load_split("train")
    base = stratified_sample(train, 1000, seed=0)
    used = {(r["utterance"], r["intent"]) for r in base}
    rest = defaultdict(list)
    for r in train:
        if (r["utterance"], r["intent"]) not in used:
            rest[r["intent"]].append(r)
    rng = random.Random(0)
    extra = []
    for intent in sorted(rest):
        rng.shuffle(rest[intent])
        extra += rest[intent][:80]
    rows = base + extra
    rng.shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text("".join(json.dumps(to_chat_example(r, LABELS)) + "\n" for r in rows))
    shutil.copy(ROOT / "data" / "mlx_aug" / "valid.jsonl", OUT / "valid.jsonl")
    print(f"train rows: {len(rows)} ({len(base)} base + {len(extra)} extra Bitext)")


if __name__ == "__main__":
    main()
