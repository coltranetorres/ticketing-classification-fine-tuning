"""Write data/mlx_aug/{train,valid}.jsonl: the 1,000 stratified Bitext train rows PLUS Gemini-generated
augmentation (data/aug/augmented.jsonl, capped per intent). Valid = Bitext val rows 0-199 plus 100 held-out
augmented rows (never in train). Holdout v1/v2 text is never used here."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from h5.data import LABELS, load_split, stratified_sample
from h5.gemma import to_chat_example

OUT = Path(__file__).resolve().parents[1] / "data" / "mlx_aug"
AUG = Path(__file__).resolve().parents[1] / "data" / "aug" / "augmented.jsonl"


def write(path, rows):
    path.write_text("".join(json.dumps(to_chat_example(r, LABELS)) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=80, help="max augmented rows per intent in train")
    args = ap.parse_args()
    aug = [json.loads(l) for l in AUG.read_text().splitlines()]
    by = defaultdict(list)
    for r in aug:
        by[r["intent"]].append(r)
    rng = random.Random(0)
    train_aug, valid_aug = [], []
    for intent in sorted(by):
        rows = by[intent][:]
        rng.shuffle(rows)
        valid_aug += rows[:4]  # ~108 total held out for the val loss
        train_aug += rows[4 : 4 + args.cap]
    base = stratified_sample(load_split("train"), 1000, seed=0)
    train = base + train_aug
    rng.shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "train.jsonl", train)
    write(OUT / "valid.jsonl", load_split("val")[:200] + valid_aug)
    print(f"train: {len(base)} bitext + {len(train_aug)} augmented = {len(train)}; valid: 200 + {len(valid_aug)}")


if __name__ == "__main__":
    main()
