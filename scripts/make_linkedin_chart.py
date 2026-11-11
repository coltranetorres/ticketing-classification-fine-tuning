"""Render results/linkedin_chart.png (1200x627): accuracy of Gemma-1B variants and Gemini on three test sets."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [  # name, color, [bitext test, fresh holdout v2, real tweets (clear)]
    ("Gemma-1B, untuned", "#2a78d6", [52.7, 34.1, 50.5]),
    ("Gemma-1B tuned on Bitext only", "#eb6834", [99.3, 67.4, 33.9]),
    ("Gemma-1B tuned on Bitext + varied data", "#1baf7a", [98.7, 90.4, 88.1]),
    ("Gemini 3.5 Flash Lite (larger, untuned)", "#8a8984", [94.5, 97.8, 96.3]),
]
GROUPS = ["Same style as training\n(Bitext test, 818 msgs)", "Worded differently\n(fresh holdout, 270 msgs)", "Real customer tweets\n(109 labeled msgs)"]

fig, ax = plt.subplots(figsize=(12, 6.27), dpi=100)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.06, right=0.97, top=0.70, bottom=0.17)
w, gap = 0.2, 0.012
for gi in range(3):
    for si, (name, color, vals) in enumerate(SERIES):
        x = gi + (si - 1.5) * (w + gap)
        ax.bar(x, vals[gi], width=w, color=color, zorder=3)
        ax.text(x, vals[gi] + 1.3, f"{vals[gi]:.0f}%", ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(GROUPS, fontsize=11.5, color=INK)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=10, color=INK2)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
# call-out on the failure case (orange bar in the tweets group)
xo = 2 + (-0.5) * (w + gap)
ax.plot([xo, xo], [43, 57], color=INK2, lw=1, zorder=2)
ax.text(1.99, 58.5, "Below untuned:\nlearned the\ndataset's style", fontsize=10.5, color=INK, ha="right", va="bottom", linespacing=1.25)
fig.text(0.06, 0.93, "The training data decides what fine-tuning does for a small model", fontsize=21, fontweight="bold", color=INK, va="top")
fig.text(0.06, 0.865, "Intent classification, 27 labels. Accuracy by test set.", fontsize=13, color=INK2, va="top")
# legend row (identity never color-alone: labels are written out)
x0 = 0.06
for name, color, _ in SERIES:
    fig.patches.append(plt.Rectangle((x0, 0.755), 0.012, 0.022, transform=fig.transFigure, color=color))
    t = fig.text(x0 + 0.017, 0.766, name, fontsize=10.2, color=INK2, va="center")
    fig.canvas.draw()
    x0 += 0.017 + t.get_window_extent().width / fig.bbox.width + 0.018
fig.text(0.06, 0.035, "Gemma-3-1B + LoRA. One training run per setting. Real-tweet labels made by an LLM with a manual spot-check. Gemini ran with reasoning on.",
         fontsize=9, color=INK2, va="bottom")
fig.savefig("results/linkedin_chart.png", facecolor=SURFACE)
