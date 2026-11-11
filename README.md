# Fine-tuning a small language model for intent classification

## Research question
**To what extent can fine-tuning improve a small language model (SLM) on intent classification, and what drives the improvement?**

1. How much does fine-tuning improve an SLM over its untuned self?
2. Does the improvement carry over to differently-worded and real messages?
3. What drives it: amount of training data, or variety?

- Task: 27 customer-service intents (Bitext dataset), custom labels
- SLM: Gemma-3-1B + LoRA, tuned on 1,000+ examples
- Yardstick: a larger untuned model, Gemini 3.5 Flash Lite (label names, or names + one-line definitions)

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

Details: [`results/report_ood.md`](results/report_ood.md) (all test sets, intervals, paired tests); [`results/report.md`](results/report.md) (original Bitext-only experiment).

## Conclusion
- **Extent: large, but conditional on the training data.** The same Gemma-1B improved from 52.7% to 98.7% on Bitext, from 34.1% to 90.4% on a fresh holdout, and from 50.5% to 88.1% on real tweets (+38 to +56 points).
- **It can also hurt.** Tuned on Bitext alone, accuracy on real tweets fell to 33.9%, below untuned: the model overfit to the dataset's style. Bitext-only gains are an easy-test result.
- **Variety, not quantity, drives the improvement.** 3x more Bitext data moved holdout v2 by under 1 point; varied training data moved it +23 (the 1,000 Bitext examples cover only 63% of the holdout's words, vs 97% with augmentation).
- **Against the larger model:** the tuned SLM beats untuned Gemini on Bitext (99% vs 94%, an easy templated test) but still trails it on real data: about 8 points on the 109 clear tweets (88% vs 96%), about 12 on all 283 (69% vs 81%). Tuning narrowed that gap from 62 points to 8.
- **Open levers (untried):** a larger base model, real-ticket training data, a confidence router that escalates uncertain tickets, more seeds, a sweep over the number of examples.
- **Limits:** one training run per setting; only 1,000 and 3,160 examples tried; real-tweet labels are LLM-made with a 50-row spot-check; the real set is small (109), complaint-heavy and covers 23 of 27 intents; Gemini ran with reasoning on; augmentation data comes from Gemini.
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
