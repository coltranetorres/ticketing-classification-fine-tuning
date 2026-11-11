"""Bitext customer-service data loading and stratified sampling."""
from __future__ import annotations

import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "archive.zip"

_FILES = {
    "train": "Bitext_Sample_Customer_Service_Training_Dataset.csv",
    "val": "Bitext_Sample_Customer_Service_Validation_Dataset.csv",
    "test": "Bitext_Sample_Customer_Service_Testing_Dataset.csv",
}


class Row(TypedDict):
    utterance: str
    intent: str


HOLDOUT_PATH = DATA_DIR / "holdout" / "synthetic_holdout.jsonl"


def load_holdout(version: int = 1) -> list[Row]:
    """Hand-written out-of-distribution sets (scripts/build_synthetic_holdout*.py). v1 = dev, v2 = fresh test."""
    import json

    path = HOLDOUT_PATH if version == 1 else HOLDOUT_PATH.with_name(f"synthetic_holdout_v{version}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [Row(utterance=d["utterance"], intent=d["intent"]) for d in map(json.loads, f)]


def load_twitter() -> list[Row]:
    """Real Twitter customer tweets labeled by Claude (clear + borderline; filter by 'confidence' in the file)."""
    import json

    path = DATA_DIR / "twitter" / "eval_all.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Row(utterance=d["utterance"], intent=d["intent"]) for d in map(json.loads, f)]


def load_split(name: Literal["train", "val", "test"]) -> list[Row]:
    path = DATA_DIR / _FILES[name]
    if not path.exists():
        with zipfile.ZipFile(ARCHIVE) as z:
            for fname in _FILES.values():
                if not (DATA_DIR / fname).exists():
                    z.extract(fname, DATA_DIR)
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Row(utterance=r["utterance"], intent=r["intent"])
            for r in csv.DictReader(f)
        ]


LABELS: list[str] = sorted({r["intent"] for r in load_split("train")})


def stratified_sample(rows: list[Row], n: int, seed: int = 0) -> list[Row]:
    """Sample n rows, balanced across labels (train rows only, by caller contract).

    Per-label quota is n // L with the remainder spread one each over labels;
    quotas are capped at availability and any shortfall is redistributed to
    labels with spare rows. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_label[r["intent"]].append(r)
    labels = sorted(by_label)
    n = min(n, len(rows))
    base, rem = divmod(n, len(labels))
    extra = set(rng.sample(labels, rem))
    quota = {
        l: min(base + (l in extra), len(by_label[l])) for l in labels
    }
    short = n - sum(quota.values())
    while short > 0:
        spare = [l for l in labels if quota[l] < len(by_label[l])]
        for l in rng.sample(spare, min(short, len(spare))):
            quota[l] += 1
            short -= 1
    out: list[Row] = []
    for l in labels:
        out.extend(rng.sample(by_label[l], quota[l]))
    return out
