# Out-of-distribution results

Accuracy with 95% Wilson interval. Tweet rows need the local (gitignored) tweet data; tweet text is never written here.

| Run | Bitext test (818) | Holdout v2 (270) | Real tweets, clear (109) | Real tweets, clear+borderline (283) |
|---|---|---|---|---|
| Gemma-1B untuned | 52.7% (49%-56%) | 34.1% (29%-40%) | 50.5% (41%-60%) | 34.6% (29%-40%) |
| Gemma-1B tuned, Bitext 1k | 99.3% (98%-100%) | 67.4% (62%-73%) | 33.9% (26%-43%) | 27.9% (23%-33%) |
| Gemma-1B tuned, Bitext 3.2k (ablation) | 99.5% (99%-100%) | 68.1% (62%-73%) | 33.9% (26%-43%) | 32.2% (27%-38%) |
| Gemma-1B tuned, Bitext + augmentation | 98.7% (98%-99%) | 90.4% (86%-93%) | 88.1% (81%-93%) | 68.9% (63%-74%) |
| Gemini Flash Lite, names | 94.0% (92%-95%) | 98.9% (97%-100%) | 89.9% (83%-94%) | 78.1% (73%-83%) |
| Gemini Flash Lite, definitions | 94.5% (93%-96%) | 97.8% (95%-99%) | 96.3% (91%-99%) | 80.9% (76%-85%) |

## Paired comparison: augmented Gemma vs Gemini (definitions)

| Set | only Gemma right | only Gemini right | exact McNemar p |
|---|---|---|---|
| Bitext test | 38 | 4 | 5.7e-08 |
| Holdout v2 | 5 | 25 | 0.00032 |
| Real tweets (all 283) | 23 | 57 | 0.00018 |

## Notes
- Holdout v2: 270 messages written by Claude, frozen before the augmented model was trained.
- Real tweets: first-contact customer tweets to 7 retail brands, labeled blind by a separate Claude agent, 50 spot-checked by the author with no corrections. 64% of the clear set is `complaint`.
- The 3.2k-example ablation matches the augmented run in size but uses Bitext only: variety, not quantity, drove the gain.
- Each configuration was trained once (single seed).
