# H5: LoRA-tuned Gemma-3-1B vs untuned Gemini on intent classification

Does a LoRA-tuned Gemma-3-1B (1,000 training examples) beat untuned Gemini 3.5 Flash Lite
on 27-way intent classification (Bitext customer-service data, 818 test rows)?

Headline: yes. Tuned Gemma scores 0.993 accuracy vs 0.940 / 0.945 for Gemini (names /
definitions prompts); untuned Gemma is far lower. Full tables, McNemar tests, confusions and
caveats (read them, esp. templated data and forced Gemini reasoning): `results/report.md`.
Chart: `results/comparison.png`. LR/checkpoint selection history: `results/gemma_selection.json`.

## Requirements
- Apple Silicon Mac (MLX training and inference), Python via [uv](https://docs.astral.sh/uv/)
- `OPENROUTER_API_KEY` in `.env` (Gemini calls go through OpenRouter)
- Gemini outputs are not bit-reproducible (API sampling); cached responses are in `results/cache_*.jsonl`

## Data
Bitext Sample Customer Service dataset (train / validation / test CSVs), expected at
`data/archive.zip`. It is NOT committed. `h5.data` unzips it on import.

## Reproduce
```bash
uv sync
uv run python scripts/prepare_mlx_data.py
uv run python scripts/run_gemini.py --variant names
uv run python scripts/run_gemini.py --variant definitions
uv run python scripts/run_gemma.py --name untuned --split test

# LoRA training: scripts/train_lora.sh [LEARNING_RATE] [ADAPTER_DIR] (750 iters, save every 250)
scripts/train_lora.sh 5e-6 adapters/lr_5e-6
scripts/train_lora.sh 1e-5 adapters
scripts/train_lora.sh 5e-5 adapters/lr_5e-5

# Select learning rate / checkpoint on the 200-row validation slice
uv run python scripts/eval_checkpoints.py

# Clean adapter dir from the chosen checkpoint (lr 5e-5, step 500)
mkdir -p adapters/final
cp adapters/lr_5e-5/0000500_adapters.safetensors adapters/final/adapters.safetensors
cp adapters/lr_5e-5/adapter_config.json adapters/final/

# Single test run of the tuned model, then the report
uv run python scripts/run_gemma.py --name tuned --adapter adapters/final --split test
uv run python scripts/report.py
```
Tests: `uv run pytest -q`.
