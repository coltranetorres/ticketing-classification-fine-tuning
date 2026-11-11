"""Run Gemma (MLX). Usage: --name {untuned,tuned} [--adapter PATH] [--split {val200,test}]"""
import argparse
import json
import time
from pathlib import Path

from mlx_lm import load

from h5.data import LABELS, load_holdout, load_split, load_twitter
from h5.gemma import predict, predict_constrained
from h5.metrics import score

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL = "mlx-community/gemma-3-1b-it-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["untuned", "tuned"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["val200", "test", "holdout", "holdout2", "twitter"], default="test")
    ap.add_argument("--constrained", action="store_true", help="decode restricted to valid labels")
    ap.add_argument("--tag", default="", help="extra suffix for the results file name")
    args = ap.parse_args()
    if args.split == "twitter":
        rows = load_twitter()
    elif args.split in ("holdout", "holdout2"):
        rows = load_holdout(2 if args.split == "holdout2" else 1)
    else:
        rows = load_split("test") if args.split == "test" else load_split("val")[:200]
    model, tok = load(MODEL, adapter_path=args.adapter)
    t0 = time.time()
    raw, confs = [], None
    if args.constrained:
        scored = predict_constrained(model, tok, [r["utterance"] for r in rows], LABELS)
        preds, confs = [l for l, _ in scored], [c for _, c in scored]
        raw = preds
    else:
        preds = predict(model, tok, [r["utterance"] for r in rows], LABELS, raw)
    print(f"predict took {time.time() - t0:.0f}s")
    metrics = score([r["intent"] for r in rows], preds, LABELS)
    out = {"metrics": metrics,
           "predictions": [{"utterance": r["utterance"], "true": r["intent"], "pred": p,
                            **({"logp": confs[i]} if confs else {})} for i, (r, p) in enumerate(zip(rows, preds))]}
    RESULTS.mkdir(exist_ok=True)
    suffix = {"val200": "_val200", "holdout": "_holdout", "holdout2": "_holdout2", "twitter": "_twitter"}.get(args.split, "")
    if args.constrained:
        suffix += "_constrained"
    suffix += args.tag
    (RESULTS / f"gemma_{args.name}{suffix}.json").write_text(json.dumps(out, indent=2))
    print(args.name, args.split, metrics)
    none = [(r["utterance"], t) for r, t, p in zip(rows, raw, preds) if p is None]
    print("None count:", len(none))
    for u, t in none[:8]:
        print(repr(u), "->", repr(t))


if __name__ == "__main__":
    main()
