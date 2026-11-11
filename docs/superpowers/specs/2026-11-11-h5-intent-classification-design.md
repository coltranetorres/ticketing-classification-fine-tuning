# H5: Tuned small model vs untuned larger model — intent classification

## Hypothesis
Gemma-3-1B, LoRA fine-tuned on 1,000 examples, beats Gemini 3.5 Flash Lite (no tuning) on 27-way intent classification.

## Data
Bitext customer-service CSVs in `data/archive.zip` (train 6,539 / val 818 / test 818). Columns: utterance, intent (27 classes, the label), category, tags (ignored). Unzip into `data/`.
- Train subset: stratified sample of 1,000 from train (~37/label), fixed seed.
- Checkpoint selection: 200-row slice of validation. Test set (818) used once at the end.

## Models
- Small: `mlx-community/gemma-3-1b-it-bf16`, LoRA via `mlx-lm` (builder's call; 4-bit QLoRA only if needed).
- Large: `google/gemini-3.5-flash-lite` via OpenRouter, OpenAI client, `load_dotenv()` for `OPENROUTER_API_KEY` from `.env`. Reasoning off by default; one reasoning-on run only if baseline looks suspiciously weak.

## Runs (all on same 818 test rows, same prompt template, output constrained to a label)
1. Gemma-1B untuned (floor)
2. Gemini zero-shot, label names only
3. Gemini zero-shot, one-line definition per label
4. Gemma-1B LoRA-tuned on 1,000 examples

## Metrics
Accuracy, macro-F1, accuracy on confusable pairs (get_invoice/check_invoice, contact_customer_service/contact_human_agent, cancel_order/check_cancellation_fee, get_refund/track_refund), confusion matrix, one comparison chart.

## Constraints
- Runtime: `uv` with `pyproject.toml` (system Python is 3.9; uv provides newer). Deps: mlx-lm, openai, python-dotenv, scikit-learn, matplotlib.
- MacBook M4, 24GB RAM.
- Never print or log the API key.

## Non-goals
Other datasets, other models, hyperparameter sweeps beyond a small LR check, N-sweep (possible follow-up).

## Anti-goals
Tuning the setup until H5 passes. If Gemini wins, report H5 as failed.
