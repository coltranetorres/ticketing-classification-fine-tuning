import json
import subprocess
import sys
from pathlib import Path

from h5.data import LABELS
from h5.gemma import to_chat_example

ROOT = Path(__file__).resolve().parents[1]


def test_to_chat_example():
    row = {"utterance": "where is my order", "intent": "track_order"}
    ex = to_chat_example(row, LABELS)
    u, a = ex["messages"]
    assert u["role"] == "user" and "where is my order" in u["content"]
    assert a == {"role": "assistant", "content": "track_order"}


def _read(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def test_prepared_files():
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_mlx_data.py")], check=True)
    train = _read(ROOT / "data/mlx/train.jsonl")
    valid = _read(ROOT / "data/mlx/valid.jsonl")
    assert len(train) == 1000 and len(valid) == 200

    def pair(ex):
        u = ex["messages"][0]["content"]
        return u.split("Customer message: ")[1].split("\n\nAnswer")[0], ex["messages"][1]["content"]

    from h5.data import load_split
    other = {r["utterance"]: r["intent"] for s in ("test",) for r in load_split(s)}
    other.update({pair(e)[0]: pair(e)[1] for e in valid})
    for e in train:
        u, l = pair(e)
        assert other.get(u, l) == l
