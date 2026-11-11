# Small tuned model vs larger untuned model: intent classification

## Research question
**H5:** Does a small model fine-tuned on a number of examples beat a larger model with no tuning, on classification with custom labels?

- Task: 27 customer-service intents (Bitext dataset)
- Small model: Gemma-3-1B + LoRA, tuned on 1,000 examples
- Larger model: Gemini 3.5 Flash Lite, untuned (label names, or names + one-line definitions)
- Follow-ups: does it hold on differently-worded and real messages? What closes the gap?

## Methodology
- **Data:** Bitext train / val / test (6,539 / 818 / 818). 1,000 stratified train rows for tuning.
- **Same prompt, same parser** for every model; unparseable output counts as wrong. Selection (learning rate, checkpoint) on dev data only; each test set scored once per model.
- **Out-of-distribution tests**, none overlapping Bitext text:
  - Holdout v1 (dev) and v2 (fresh, frozen before new training): 270 messages each, hand-written by Claude, 10 per intent.
  - **Real tweets:** Customer Support on Twitter. First-contact tweets to 7 retail brands (chosen before looking at content), 600 sampled, labeled blind by a separate Claude agent ("none" allowed), 50 spot-checked by the author (0 corrections). Evaluated on the 109 clearly-labeled tweets (283 incl. borderline).
- **Augmentation:** Gemini generated about 3,000 varied messages (6 styles per intent) from label names and definitions only, kept only if Gemini labeled them back to the intended intent. Training set: 1,000 Bitext + 2,160 generated.
- **Ablation:** 3,160 Bitext-only examples (same size, no new variety).
- **Constrained decoding:** the model can only output one of the 27 labels.

## Results (accuracy)

| Model | Bitext test | Holdout v2 | Real tweets (n=109) |
|---|---|---|---|
| Gemma-1B untuned | 52.7% | 34.1% | 50.5% |
| Gemma-1B tuned, Bitext 1k | **99.3%** | 67.4% | 33.9% |
| Gemma-1B tuned, Bitext 3.2k (ablation) | 99.5% | 68.1% | 33.9% |
| Gemma-1B tuned, Bitext + augmentation | 98.7% | 90.4% | 88.1% |
| Gemini Flash Lite, names | 94.0% | 98.9% | 89.9% |
| Gemini Flash Lite, definitions | 94.5% | 97.8% | **96.3%** |

- Bitext test: tuned beats Gemini (McNemar p ≈ 1e-9).
- **Variety, not quantity, drove the gain:** 3x more Bitext data moved holdout v2 by under 1 point; adding varied data moved it +23. Bitext-1k covers 63% of the words in v2; with augmentation, 97%.
- **Bitext-only tuning hurts on real tweets** (33.9%, below untuned). It labels most complaints as `delivery_period`. Augmentation fixes this.
- Real tweets are 64% `complaint`; always guessing it scores 64%. 95% CI for the augmented model: 81-93%; Gemini definitions: 91-99%.
- On non-complaint real tweets (n=39), augmented Gemma ties Gemini definitions (95%); on complaints it trails (84% vs 97%).

Details: [`results/report.md`](results/report.md) (Bitext experiment).

## Conclusion
- **H5 holds in-distribution:** tuning a 1B model on 1,000 examples beat untuned Gemini on Bitext.
- **H5 does not hold out-of-distribution:** with Bitext-only tuning it loses to Gemini on new wording and collapses on real tweets. The 99% mostly reflects learning Bitext's wording.
- **Diverse training data is the fix:** it brings the small model within about 8 points of Gemini on real tweets, but does not beat it.
- **Limits:** one training run per setting; labels for real tweets are LLM-made with a 50-row spot-check; the real set is small (109), complaint-heavy and covers 23 of 27 intents; Gemini ran with reasoning on; augmentation data comes from Gemini.
- **Not shown:** cost savings or CPU hosting.

## Reproduce
Needs an Apple Silicon Mac (MLX), [uv](https://docs.astral.sh/uv/), `OPENROUTER_API_KEY` in `.env`, `data/archive.zip` (Bitext Sample Customer Service dataset) and, for the tweet test, `data/twitter/twcs/twcs.csv` from Kaggle. Neither dataset is committed. Gemini outputs are not bit-reproducible.

```bash
uv sync && uv run pytest -q
uv run python scripts/prepare_mlx_data.py                         # 1,000 Bitext rows
scripts/train_lora.sh 5e-5 adapters/lr_5e-5                        # Bitext-only model
uv run python scripts/build_synthetic_holdout.py                   # holdout v1 (dev)
uv run python scripts/build_synthetic_holdout_v2.py                # holdout v2
uv run python scripts/generate_augmentation.py                     # needs OpenRouter key
uv run python scripts/prepare_mlx_data_aug.py
scripts/train_lora.sh 5e-5 adapters/aug_lr5e-5 data/mlx_aug 1500
uv run python scripts/eval_checkpoints_dev.py adapters/aug_lr5e-5 results/aug_dev.json   # pick checkpoint on dev only
uv run python scripts/prepare_mlx_data_bitext3k.py                 # ablation data
uv run python scripts/sample_twitter.py                            # unlabeled candidate tweets
uv run python scripts/run_gemma.py --name tuned --adapter <dir> --split holdout2 --constrained
uv run python scripts/run_gemma.py --name tuned --adapter <dir> --split twitter --constrained
uv run python scripts/run_gemini.py --variant names --dataset twitter
```
Each adapter dir needs `adapter_config.json` + the chosen checkpoint copied as `adapters.safetensors`. Tweet labels (`data/twitter/labels_claude.jsonl`, `eval_all.jsonl`) are not in the repo.
