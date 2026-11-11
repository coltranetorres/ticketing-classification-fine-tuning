"""Write data/mlx/train.jsonl (1000 stratified train) and valid.jsonl (first 200 val rows)."""
import json
from pathlib import Path

from h5.data import LABELS, load_split, stratified_sample
from h5.gemma import to_chat_example

OUT = Path(__file__).resolve().parents[1] / "data" / "mlx"


def write(path, rows):
    path.write_text("".join(json.dumps(to_chat_example(r, LABELS)) + "\n" for r in rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "train.jsonl", stratified_sample(load_split("train"), 1000, seed=0))
    write(OUT / "valid.jsonl", load_split("val")[:200])


if __name__ == "__main__":
    main()
