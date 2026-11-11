"""Remove Bitext utterance text from committed results files (keep true label, prediction, order).
Run before committing results produced from Bitext splits. Holdout/tweet result files are left alone."""
import json
import sys
from pathlib import Path

SKIP = ("holdout", "twitter", "cache_")


def main() -> None:
    for p in sorted(Path("results").glob("*.json")):
        if any(s in p.name for s in SKIP):
            continue
        d = json.loads(p.read_text())
        rows = d.get("predictions") if isinstance(d, dict) else None
        if not rows or "utterance" not in rows[0]:
            continue
        for r in rows:
            r.pop("utterance", None)
        p.write_text(json.dumps(d, indent=2))
        print("stripped", p, len(rows), "rows")


if __name__ == "__main__":
    main()
