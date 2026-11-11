"""Aggregate report across Bitext test, holdout v2 and real tweets -> results/report_ood.md (numbers only, no tweet text)."""
import json
import math
from pathlib import Path

R = Path("results")
TW = Path("data/twitter/eval_all.jsonl")

RUNS = {  # name -> {set: results file stem}
    "Gemma-1B untuned": {"test": "gemma_untuned", "v2": "gemma_untuned_holdout2_constrained", "tw": "gemma_untuned_twitter_constrained"},
    "Gemma-1B tuned, Bitext 1k": {"test": "gemma_tuned", "v2": "gemma_tuned_holdout2_constrained_bitext_only", "tw": "gemma_tuned_twitter_constrained_bitext_only"},
    "Gemma-1B tuned, Bitext 3.2k (ablation)": {"test": "gemma_tuned_constrained_bitext3k", "v2": "gemma_tuned_holdout2_constrained_bitext3k", "tw": "gemma_tuned_twitter_constrained_bitext3k"},
    "Gemma-1B tuned, Bitext + augmentation": {"test": "gemma_tuned_constrained_aug", "v2": "gemma_tuned_holdout2_constrained_aug", "tw": "gemma_tuned_twitter_constrained_aug"},
    "Gemini Flash Lite, names": {"test": "gemini_names", "v2": "gemini_names_holdout2", "tw": "gemini_names_twitter"},
    "Gemini Flash Lite, definitions": {"test": "gemini_definitions", "v2": "gemini_definitions_holdout2", "tw": "gemini_definitions_twitter"},
}


def preds(stem):
    p = R / f"{stem}.json"
    return json.load(open(p))["predictions"] if p.exists() else None


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cell(rows):
    if not rows:
        return "n/a"
    k = sum(r["pred"] == r["true"] for r in rows)
    lo, hi = wilson(k, len(rows))
    return f"{k / len(rows):.1%} ({lo:.0%}-{hi:.0%})"


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def paired(a, g):
    b = sum(x["pred"] == x["true"] and y["pred"] != y["true"] for x, y in zip(a, g))
    c = sum(x["pred"] != x["true"] and y["pred"] == y["true"] for x, y in zip(a, g))
    return b, c, mcnemar_p(b, c)


def main():
    conf = {}
    if TW.exists():
        for l in open(TW):
            d = json.loads(l)
            conf[d["utterance"]] = d["confidence"]
    L = ["# Out-of-distribution results", "",
         "Accuracy with 95% Wilson interval. Tweet rows need the local (gitignored) tweet data; "
         "tweet text is never written here.", "",
         "| Run | Bitext test (818) | Holdout v2 (270) | Real tweets, clear (109) | Real tweets, clear+borderline (283) |",
         "|---|---|---|---|---|"]
    cache = {}
    for name, files in RUNS.items():
        cells = []
        for key in ("test", "v2", "tw"):
            P = preds(files[key])
            cache[(name, key)] = P
            if key == "tw":
                clear = [r for r in P if conf.get(r["utterance"]) == "clear"] if P and conf else None
                cells += [cell(clear), cell(P)]
            else:
                cells.append(cell(P))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    aug, gd = "Gemma-1B tuned, Bitext + augmentation", "Gemini Flash Lite, definitions"
    L += ["", "## Paired comparison: augmented Gemma vs Gemini (definitions)", "",
          "| Set | only Gemma right | only Gemini right | exact McNemar p |", "|---|---|---|---|"]
    for key, label in (("test", "Bitext test"), ("v2", "Holdout v2"), ("tw", "Real tweets (all 283)")):
        a, g = cache[(aug, key)], cache[(gd, key)]
        if a and g:
            b, c, p = paired(a, g)
            L.append(f"| {label} | {b} | {c} | {p:.2g} |")
    L += ["", "## Notes",
          "- Holdout v2: 270 messages written by Claude, frozen before the augmented model was trained.",
          "- Real tweets: first-contact customer tweets to 7 retail brands, labeled blind by a separate Claude agent, "
          "50 spot-checked by the author with no corrections. 64% of the clear set is `complaint`.",
          "- The 3.2k-example ablation matches the augmented run in size but uses Bitext only: variety, not quantity, drove the gain.",
          "- Each configuration was trained once (single seed)."]
    (R / "report_ood.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
