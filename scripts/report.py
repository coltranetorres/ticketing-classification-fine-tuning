"""Build results/report.md and results/comparison.png from the four run files."""
import json
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = [
    ("gemma_untuned", "Gemma-3-1B untuned"),
    ("gemini_names", "Gemini (label names)"),
    ("gemini_definitions", "Gemini (definitions)"),
    ("gemma_tuned", "Gemma-3-1B LoRA-tuned"),
]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def top_confusions(rows: list[dict], k: int = 10) -> list[tuple[str, str, int]]:
    counts = Counter(
        (r["true"], "(unparseable)" if r["pred"] is None else r["pred"])
        for r in rows if r["true"] != r["pred"])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [(t, p, n) for (t, p), n in ranked]


def paired_counts(tuned: list[dict], other: list[dict]) -> tuple[int, int]:
    """(rows only tuned right, rows only other right); rows must align."""
    only_t = only_o = 0
    for a, b in zip(tuned, other):
        assert a["utterance"] == b["utterance"] if "utterance" in a and "utterance" in b else True
        ta, tb = a["true"] == a["pred"], b["true"] == b["pred"]
        only_t += ta and not tb
        only_o += tb and not ta
    return only_t, only_o


def verdict(tuned: dict, names: dict, defs: dict) -> str:
    if tuned["accuracy"] > names["accuracy"] and tuned["accuracy"] > defs["accuracy"]:
        return (f"H5 supported: tuned Gemma accuracy {tuned['accuracy']:.4f} exceeds both "
                f"Gemini variants ({names['accuracy']:.4f}, {defs['accuracy']:.4f}).")
    return (f"H5 not supported: tuned Gemma accuracy {tuned['accuracy']:.4f} does not exceed "
            f"both Gemini variants ({names['accuracy']:.4f}, {defs['accuracy']:.4f}).")


def make_chart(data: dict, path: Path) -> None:
    labels = [name for _, name in RUNS]
    acc = [data[k]["metrics"]["accuracy"] for k, _ in RUNS]
    f1 = [data[k]["metrics"]["macro_f1"] for k, _ in RUNS]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    w = 0.38
    xs = range(len(labels))
    for off, vals, name, color in [(-w / 2, acc, "Accuracy", "#2a6fb0"),
                                   (w / 2, f1, "Macro-F1", "#e08a1e")]:
        bars = ax.bar([x + off for x in xs], vals, w, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, color="#222")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.replace(" (", "\n(").replace(" LoRA", "\nLoRA") for l in labels],
                       fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([i / 10 for i in range(0, 11, 2)])
    ax.set_ylabel("Score (0 to 1)")
    ax.set_title("27-way intent classification, 818 test rows", fontsize=13, pad=30)
    ax.grid(axis="y", color="#ddd")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_confusions(rows: list[dict]) -> str:
    out = ["| True | Predicted | Count |", "|---|---|---|"]
    out += [f"| {t} | {p} | {n} |" for t, p, n in top_confusions(rows)]
    return "\n".join(out)


CAVEATS = """\
- Gemini ran with its default reasoning ON, which was forced: OpenRouter returned a 400 error when reasoning was disabled for this model, although the spec asked for reasoning OFF. Reasoning gives Gemini extra inference-time compute that Gemma (no reasoning mode) lacks, so the comparison is not matched on compute; this favours Gemini, so the verdict is conservative on that axis.
- Gemma sees the full label list in its prompt, just like Gemini; tuning teaches the format and label boundaries, not the label set.
- The Gemini "definitions" variant uses label definitions written by us from the label names only (not from inspecting data or errors).
- The data is synthetic, templated Bitext utterances; high scores here may not transfer to real customer messages.
- The gain largely reflects learning this dataset's annotation conventions rather than general intent understanding. Gemini (definitions) top errors are create_account predicted as registration_problems (8), cancel_order as check_cancellation_fee (6) and set_up_shipping_address as change_shipping_address (5): label boundaries that are arbitrary without the dataset's conventions.
- Near-duplicate check (done once offline, not produced by report.py; test vs the 1k training subset): exact overlap is 0 between test and train (full train, 1k subset) and val200; 2 test rows match a training row exactly after normalising punctuation and case. Token-Jaccard similarity to the nearest training example: 9 test rows reach 1.0, 22 reach >=0.9, 103 (12.6%) reach >=0.8 and 300 (37%) reach >=0.7. Accuracy by similarity band (tuned Gemma vs Gemini-definitions): <0.5: 106 rows, 0.981 vs 0.906; 0.5-0.7: 412 rows, 0.993 vs 0.952; 0.7-0.8: 197 rows, 1.000 vs 0.944; >=0.8: 103 rows, 0.990 vs 0.961. The gap does not shrink on the least-similar rows, so near-duplicate leakage does not explain the result. However the data are templated and synthetic, so the absolute ~99% likely overstates performance on real customer messages.
- The test set has 818 rows, so one row is about 0.12 percentage points.
- Validation was a 200-row slice. It did separate the learning rates (step-500 scores 0.980, 0.985 and 1.000 for lr 5e-6, 1e-5 and 5e-5); only the choice of step within lr 5e-5 (step 500 at 1.000 vs 0.995 for steps 250 and 750) came down to a single row. The tuned model was run on the test set once.
- Training used only 1,000 examples (about 37 per label).
- Single seed, single run per condition: no estimate of run-to-run variance. The McNemar test covers test-row sampling only, not training or sampling randomness.
"""


def main() -> None:
    data = {k: json.load(open(RESULTS / f"{k}.json")) for k, _ in RUNS}
    tab = ["| Run | Accuracy | Macro-F1 | Confusable accuracy | Unparseable | n |",
           "|---|---|---|---|---|---|"]
    for k, name in RUNS:
        m = data[k]["metrics"]
        tab.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                   f"{m['confusable_accuracy']:.4f} | {m['unparseable']} | {m['n']} |")
    tuned = data["gemma_tuned"]
    pair = ["| Comparison | Only tuned right | Only Gemini right | McNemar p (exact, two-sided) |",
            "|---|---|---|---|"]
    for k, name in RUNS[1:3]:
        a, b = paired_counts(tuned["predictions"], data[k]["predictions"])
        pair.append(f"| Tuned Gemma vs {name} | {a} | {b} | {mcnemar_p(a, b):.4g} |")
    v = verdict(tuned["metrics"], data["gemini_names"]["metrics"],
                data["gemini_definitions"]["metrics"])
    md = f"""# H5 results: LoRA-tuned Gemma-3-1B vs untuned Gemini on 27-way intent classification

**Verdict: {v}**

![Accuracy and macro-F1 per run](comparison.png)

## Metrics

{chr(10).join(tab)}

Unparseable outputs count as wrong in every metric.

## Paired comparison (tuned Gemma vs Gemini)

{chr(10).join(pair)}

"Only tuned right" / "only Gemini right" count test rows where exactly one of the two is correct. The p-value is an exact binomial McNemar test on those discordant rows.

## Top-10 confusions: tuned Gemma

{md_confusions(tuned["predictions"])}

## Top-10 confusions: Gemini (definitions)

{md_confusions(data["gemini_definitions"]["predictions"])}

## Setup and caveats

{CAVEATS}"""
    (RESULTS / "report.md").write_text(md)
    make_chart(data, RESULTS / "comparison.png")
    print(md)


if __name__ == "__main__":
    main()
