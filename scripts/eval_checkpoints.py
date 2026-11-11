"""Evaluate every checkpoint in adapter dirs on val200 only.
Usage: eval_checkpoints.py DIR [DIR...]  -> writes results/gemma_selection.json (and
results/gemma_tuned_checkpoints.json for the dir named 'adapters')."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from mlx_lm import load

from h5.data import LABELS, load_split
from h5.gemma import predict
from h5.metrics import score

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/gemma-3-1b-it-bf16"


def main():
    rows = load_split("val")[:200]
    utts = [r["utterance"] for r in rows]
    gold = [r["intent"] for r in rows]
    sel_path = ROOT / "results" / "gemma_selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    for d in sys.argv[1:]:
        d = Path(d)
        accs = {}
        for ck in sorted(d.glob("0*_adapters.safetensors")) + [d / "adapters.safetensors"]:
            with tempfile.TemporaryDirectory() as t:
                shutil.copy(d / "adapter_config.json", t)
                shutil.copy(ck, Path(t) / "adapters.safetensors")
                model, tok = load(MODEL, adapter_path=t)
                preds = predict(model, tok, utts, LABELS)
            m = score(gold, preds, LABELS)
            accs[ck.name] = m["accuracy"] if "accuracy" in m else m
            print(d, ck.name, accs[ck.name], "none:", sum(p is None for p in preds), flush=True)
        sel[d.name] = accs
        if d.name == "adapters":
            (ROOT / "results" / "gemma_tuned_checkpoints.json").write_text(json.dumps(accs, indent=2))
    sel_path.write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
