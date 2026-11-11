"""Run Gemma (MLX). Usage: --name {untuned,tuned} [--adapter PATH] [--split {val200,test}]"""
import argparse
import json
import time
from pathlib import Path

from mlx_lm import load

from h5.data import LABELS, load_holdout, load_split
from h5.gemma import predict
from h5.metrics import score

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL = "mlx-community/gemma-3-1b-it-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["untuned", "tuned"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["val200", "test", "holdout"], default="test")
    args = ap.parse_args()
    if args.split == "holdout":
        rows = load_holdout()
    else:
        rows = load_split("test") if args.split == "test" else load_split("val")[:200]
    model, tok = load(MODEL, adapter_path=args.adapter)
    t0 = time.time()
    raw = []
    preds = predict(model, tok, [r["utterance"] for r in rows], LABELS, raw)
    print(f"predict took {time.time() - t0:.0f}s")
    metrics = score([r["intent"] for r in rows], preds, LABELS)
    out = {"metrics": metrics,
           "predictions": [{"utterance": r["utterance"], "true": r["intent"], "pred": p} for r, p in zip(rows, preds)]}
    RESULTS.mkdir(exist_ok=True)
    suffix = {"val200": "_val200", "holdout": "_holdout"}.get(args.split, "")
    (RESULTS / f"gemma_{args.name}{suffix}.json").write_text(json.dumps(out, indent=2))
    print(args.name, args.split, metrics)
    none = [(r["utterance"], t) for r, t, p in zip(rows, raw, preds) if p is None]
    print("None count:", len(none))
    for u, t in none[:8]:
        print(repr(u), "->", repr(t))


if __name__ == "__main__":
    main()
