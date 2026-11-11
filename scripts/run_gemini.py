"""Run Gemini baseline on the test split. Usage: --variant {names,definitions} [--limit N]"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from h5.data import LABELS, load_split
from h5.gemini import classify
from h5.metrics import score
from h5.prompts import LABEL_DEFINITIONS

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["names", "definitions"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    defs = LABEL_DEFINITIONS if args.variant == "definitions" else None
    rows = load_split("test")
    if args.limit:
        rows = rows[: args.limit]
    RESULTS.mkdir(exist_ok=True)
    cache_path = RESULTS / f"cache_gemini_{args.variant}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            d = json.loads(line)
            cache[d["utterance"]] = d["pred"]
    lock = threading.Lock()
    f = open(cache_path, "a")

    def work(row):
        u = row["utterance"]
        if u in cache:
            return cache[u]
        pred = classify(u, LABELS, defs, client)
        with lock:
            f.write(json.dumps({"utterance": u, "pred": pred}) + "\n")
            f.flush()
        return pred

    with ThreadPoolExecutor(8) as ex:
        preds = list(ex.map(work, rows))
    f.close()
    trues = [r["intent"] for r in rows]
    metrics = score(trues, preds, LABELS)
    out = {"metrics": metrics,
           "predictions": [{"utterance": r["utterance"], "true": r["intent"], "pred": p} for r, p in zip(rows, preds)]}
    (RESULTS / f"gemini_{args.variant}.json").write_text(json.dumps(out, indent=2))
    print(args.variant, metrics)


if __name__ == "__main__":
    main()
