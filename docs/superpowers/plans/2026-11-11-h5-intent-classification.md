# H5 Intent Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether Gemma-3-1B LoRA-tuned on 1,000 examples beats untuned Gemini 3.5 Flash Lite on 27-way intent classification.

**Architecture:** A small `h5` package holds shared pieces (data, prompt/parse, metrics). One script per run writes predictions to `results/<run>.json`; a final script reads all four result files and produces a table and chart. The same prompt template and parser are used for every run, including the training data for LoRA.

**Tech Stack:** Python >=3.11 via `uv`; `mlx-lm`, `openai`, `python-dotenv`, `scikit-learn`, `matplotlib`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-11-h5-intent-classification-design.md`

## Global Constraints

- Runtime is `uv` with `pyproject.toml`; run everything with `uv run`. System Python is 3.9, so set `requires-python = ">=3.11"`.
- Label is the Bitext `intent` column (27 classes). `category` and `tags` are ignored.
- Splits: train 6,539 / val 818 / test 818. Unzip `data/archive.zip` into `data/` (do not modify the zip).
- Train subset: stratified 1,000 from train, `seed=0`. Checkpoint selection uses the first 200 rows of val. Test (818 rows) is scored once per run, never used for selection.
- Small model: `mlx-community/gemma-3-1b-it-bf16`. Large model: `google/gemini-3.5-flash-lite` via OpenRouter, OpenAI client, `base_url="https://openrouter.ai/api/v1"`, key from `load_dotenv()` as `OPENROUTER_API_KEY`. Reasoning off by default.
- Never print or log the API key.
- Unparseable model output counts as wrong (prediction `None`), never silently remapped.
- Anti-goal: do not change the setup after seeing test results to make H5 pass. Report the outcome as it is.
- The project is not a git repo, so there are no commit steps. Run `git init` first only if the user asks.

## Review Focus

- Model output with extra text around the label ("Intent: get_invoice.", backticks, uppercase): parser should still find the label.
- Model output naming two labels or a label that is a substring of another (`get_refund` vs `track_refund`, `check_invoice` vs `get_invoice`): treat as unparseable if ambiguous, never pick the first.
- OpenRouter rate limit, timeout or empty content on a row: retry then record `None`; the run must not crash or lose finished rows (cache results per row to disk so a rerun resumes).
- Stratified sample when a label has fewer than 38 train rows: take all of them, do not raise.
- Test or val rows leaking into the 1,000-row training sample: the sampler must draw only from train, and a test asserts no utterance overlap with test.

---

### Task 1: Project scaffold and data loading

**Files:**
- Create: `pyproject.toml`, `src/h5/__init__.py`, `src/h5/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `load_split(name: Literal["train","val","test"]) -> list[Row]` where `Row = TypedDict("Row", utterance=str, intent=str)`; unzips the archive into `data/` on first call if the CSVs are missing.
  - `LABELS: list[str]` (27 sorted intents, derived from train).
  - `stratified_sample(rows: list[Row], n: int, seed: int = 0) -> list[Row]`.

- [ ] **Step 1: Write failing tests** in `tests/test_data.py`: `len(load_split("train")) == 6539`, `len(load_split("val")) == 818`, `len(load_split("test")) == 818`, `len(LABELS) == 27`; `stratified_sample(train, 1000)` has length 1000, every label has between 36 and 38 rows, is identical across two calls with the same seed; a tiny synthetic list where one label has 3 rows returns all 3 of that label without raising; no sampled utterance+intent pair appears in `load_split("test")` as the same row object source (assert the sampler's input is train only by checking `set(u for u in sample) <= set(u for u in train)`).
- [ ] **Step 2: Run** `uv run pytest tests/test_data.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Create `pyproject.toml`** with the dependencies above and `uv sync`; implement `data.py` per the interfaces (stdlib `csv`, `zipfile`). Allocation per label is `round(n/len(labels))` with the remainder spread so the total is exactly `n`, capped at availability.
- [ ] **Step 4: Run** `uv run pytest tests/test_data.py -v`. Expected: all PASS.

### Task 2: Prompt template and output parser

**Files:**
- Create: `src/h5/prompts.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `LABELS` from Task 1.
- Produces:
  - `build_prompt(utterance: str, labels: list[str], definitions: dict[str,str] | None = None) -> str` (instruction listing labels, optional `label: definition` lines, the utterance, and "Answer with exactly one label.").
  - `parse_label(text: str | None, labels: list[str]) -> str | None`.
  - `LABEL_DEFINITIONS: dict[str,str]` with one short sentence for each of the 27 labels, written from the label name and general meaning only (do not read test rows to write them).

- [ ] **Step 1: Write failing tests**: `parse_label("get_invoice", L) == "get_invoice"`; `parse_label("Intent: **Get_Invoice**.", L) == "get_invoice"`; `parse_label("`track_refund`\n", L) == "track_refund"`; `parse_label("get_refund or track_refund", L) is None`; `parse_label("check_invoice", L) == "check_invoice"` (not confused with `get_invoice`); `parse_label("", L) is None`; `parse_label(None, L) is None`; `set(LABEL_DEFINITIONS) == set(LABELS)`; `build_prompt` output contains every label and the utterance, and contains definitions only when passed.
- [ ] **Step 2: Run** `uv run pytest tests/test_prompts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Parser lowercases, finds whole-token label matches (regex with `(?<![a-z_])label(?![a-z_])`), returns the label if exactly one distinct label matches, else `None`.
- [ ] **Step 4: Run tests.** Expected: PASS.

### Task 3: Metrics

**Files:**
- Create: `src/h5/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `CONFUSABLE_PAIRS: list[tuple[str,str]]` = get_invoice/check_invoice, contact_customer_service/contact_human_agent, cancel_order/check_cancellation_fee, get_refund/track_refund.
  - `score(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict` with keys `accuracy`, `macro_f1`, `confusable_accuracy` (accuracy over rows whose true label is in any pair), `unparseable` (count of `None`), `n`.

- [ ] **Step 1: Write failing tests**: perfect predictions give accuracy 1.0 and macro_f1 1.0; a `None` prediction counts as wrong and increments `unparseable`; `confusable_accuracy` ignores rows with non-pair labels (hand-built 4-row example with known answer 0.5).
- [ ] **Step 2: Run** `uv run pytest tests/test_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with scikit-learn (`None` mapped to a sentinel label outside `labels` for F1, with `labels=labels` passed so the sentinel is excluded from the average).
- [ ] **Step 4: Run tests.** Expected: PASS.

### Task 4: Gemini baseline (names-only and with definitions)

**Files:**
- Create: `src/h5/gemini.py`, `scripts/run_gemini.py`, `tests/test_gemini.py`

**Interfaces:**
- Consumes: `build_prompt`, `parse_label`, `LABEL_DEFINITIONS`, `score`, `load_split`, `LABELS`.
- Produces:
  - `classify(utterance: str, labels: list[str], definitions: dict[str,str] | None, client) -> str | None` (one call, retries up to 3 times with backoff on exceptions or empty content, returns the parsed label).
  - `scripts/run_gemini.py --variant {names,definitions} [--limit N]` writes `results/gemini_<variant>.json` containing `{"metrics": {...}, "predictions": [{"utterance","true","pred"}]}`; caches per-row predictions in `results/cache_gemini_<variant>.jsonl` so a rerun skips finished rows; runs with a thread pool of 8.

- [ ] **Step 1: Write failing tests** using a fake client object whose `chat.completions.create` returns canned content: returns the parsed label for `"get_invoice"`; returns `None` after 3 failed attempts when the fake raises; retries then succeeds when the fake raises once; makes the request with `model="google/gemini-3.5-flash-lite"` and `extra_body={"reasoning": {"enabled": False}}` (assert on the recorded kwargs).
- [ ] **Step 2: Run** `uv run pytest tests/test_gemini.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `classify` and the script. The script calls `load_dotenv()` and builds the client as in the user's sample. Use `temperature=0` and small `max_tokens`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Smoke run** `uv run python scripts/run_gemini.py --variant names --limit 20`. Expected: writes the result file with 20 predictions and no crash. Check `unparseable` is low; if more than a few rows are unparseable, inspect the raw outputs before the full run.
- [ ] **Step 6: Full runs** for `names` and `definitions` on all 818 test rows. Expected: two result files with `n == 818`.

### Task 5: Gemma data prep and untuned floor

**Files:**
- Create: `src/h5/gemma.py`, `scripts/prepare_mlx_data.py`, `scripts/run_gemma.py`, `tests/test_gemma.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `to_chat_example(row: Row, labels: list[str]) -> dict` returning `{"messages":[{"role":"user","content":build_prompt(...)},{"role":"assistant","content":row["intent"]}]}`.
  - `scripts/prepare_mlx_data.py` writes `data/mlx/train.jsonl` (1,000 stratified), `data/mlx/valid.jsonl` (val rows 0–199).
  - `predict(model, tokenizer, utterances: list[str], labels: list[str]) -> list[str | None]` using `mlx_lm` `generate` with greedy decoding and `max_tokens=12`, applying the chat template.
  - `scripts/run_gemma.py --name {untuned,tuned} [--adapter PATH] [--split {val200,test}]` writes `results/gemma_<name>.json` in the same format as Task 4.

- [ ] **Step 1: Write failing tests**: `to_chat_example` has the user prompt containing the utterance and the assistant content equal to the intent; the prepared train file has 1,000 lines, valid file has 200 lines, and no train utterance equals a valid or test utterance with a different label (overlap is allowed only for identical text and label).
- [ ] **Step 2: Run** `uv run pytest tests/test_gemma.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the helpers and scripts. Load the model with `mlx_lm.load`, apply the chat template with `add_generation_prompt=True`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Run the floor**: `uv run python scripts/run_gemma.py --name untuned --split test`. Expected: `results/gemma_untuned.json` with `n == 818`. A low score is expected and fine.

### Task 6: LoRA fine-tune and tuned evaluation

**Files:**
- Create: `scripts/train_lora.sh`, `results/` output only

**Interfaces:**
- Consumes: `data/mlx/` from Task 5 and `run_gemma.py`.
- Produces: adapters in `adapters/` (checkpoints saved every 250 iters), `results/gemma_tuned.json`, `results/gemma_tuned_checkpoints.json` (val200 accuracy per checkpoint).

- [ ] **Step 1: Write `scripts/train_lora.sh`** running `uv run mlx_lm.lora --model mlx-community/gemma-3-1b-it-bf16 --train --data data/mlx --adapter-path adapters --batch-size 4 --iters 750 --save-every 250 --learning-rate 1e-5 --num-layers 16` (about 3 epochs over 1,000 rows; these are the starting values).
- [ ] **Step 2: Run training.** Expected: loss decreases and checkpoint files appear in `adapters/`. If memory errors occur, lower `--batch-size` to 2.
- [ ] **Step 3: Select checkpoint**: evaluate each saved checkpoint with `run_gemma.py --split val200` and write the accuracies to `results/gemma_tuned_checkpoints.json`. Pick the best by val200 accuracy. One learning-rate check (5e-6, 1e-5, 5e-5) is allowed, also judged on val200 only.
- [ ] **Step 4: Final test run** once with the chosen adapter: `uv run python scripts/run_gemma.py --name tuned --adapter <best> --split test`. Expected: `results/gemma_tuned.json` with `n == 818`. Do not rerun after seeing the result.

### Task 7: Report

**Files:**
- Create: `scripts/report.py`, `results/report.md`, `results/comparison.png`

**Interfaces:**
- Consumes: the four `results/*.json` files from Tasks 4–6.

- [ ] **Step 1: Implement `scripts/report.py`** producing a markdown table (run, accuracy, macro-F1, confusable accuracy, unparseable), a bar chart of accuracy and macro-F1 per run, the top 10 confusion pairs for the tuned model and for Gemini-definitions, and a one-line H5 verdict: "H5 supported" if tuned accuracy exceeds both Gemini variants, else "H5 not supported". Include a McNemar-style count of rows where only the tuned model is right versus only Gemini is right.
- [ ] **Step 2: Run** `uv run python scripts/report.py`. Expected: `results/report.md` and `results/comparison.png` exist, the table lists all four runs with `n = 818`.
- [ ] **Step 3: Run the full suite** `uv run pytest -v`. Expected: all PASS.
