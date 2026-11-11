"""Score every checkpoint of one adapter dir on DEV data only (Bitext val200 + holdout v1), constrained decoding.
Usage: eval_checkpoints_dev.py ADAPTER_DIR OUT_JSON   (never touches Bitext test or holdout v2)"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from mlx_lm import load

from h5.data import LABELS, load_holdout, load_split
from h5.gemma import predict_constrained

MODEL = "mlx-community/gemma-3-1b-it-bf16"


def acc(model, tok, rows):
    preds = predict_constrained(model, tok, [r["utterance"] for r in rows], LABELS)
    return sum(p == r["intent"] for (p, _), r in zip(preds, rows)) / len(rows)


def main():
    d = Path(sys.argv[1])
    val, dev = load_split("val")[:200], load_holdout(1)
    res = {}
    for ck in sorted(d.glob("0*_adapters.safetensors")):
        with tempfile.TemporaryDirectory() as t:
            shutil.copy(d / "adapter_config.json", t)
            shutil.copy(ck, Path(t) / "adapters.safetensors")
            model, tok = load(MODEL, adapter_path=t)
            res[ck.name] = {"val200": acc(model, tok, val), "dev_v1": acc(model, tok, dev)}
            print(ck.name, res[ck.name], flush=True)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
