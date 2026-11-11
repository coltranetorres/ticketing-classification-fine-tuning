# H5 results: LoRA-tuned Gemma-3-1B vs untuned Gemini on 27-way intent classification

**Verdict: H5 supported: tuned Gemma accuracy 0.9927 exceeds both Gemini variants (0.9401, 0.9450).**

![Accuracy and macro-F1 per run](comparison.png)

## Metrics

| Run | Accuracy | Macro-F1 | Confusable accuracy | Unparseable | n |
|---|---|---|---|---|---|
| Gemma-3-1B untuned | 0.5269 | 0.5057 | 0.4936 | 74 | 818 |
| Gemini (label names) | 0.9401 | 0.9377 | 0.9191 | 0 | 818 |
| Gemini (definitions) | 0.9450 | 0.9416 | 0.9277 | 0 | 818 |
| Gemma-3-1B LoRA-tuned | 0.9927 | 0.9936 | 0.9872 | 1 | 818 |

Unparseable outputs count as wrong in every metric.

## Paired comparison (tuned Gemma vs Gemini)

| Comparison | Only tuned right | Only Gemini right | McNemar p (exact, two-sided) |
|---|---|---|---|
| Tuned Gemma vs Gemini (label names) | 48 | 5 | 7.077e-10 |
| Tuned Gemma vs Gemini (definitions) | 43 | 4 | 2.781e-09 |

"Only tuned right" / "only Gemini right" count test rows where exactly one of the two is correct. The p-value is an exact binomial McNemar test on those discordant rows.

## Top-10 confusions: tuned Gemma

| True | Predicted | Count |
|---|---|---|
| get_invoice | check_invoice | 3 |
| delete_account | (unparseable) | 1 |
| set_up_shipping_address | change_shipping_address | 1 |
| track_order | delivery_period | 1 |

## Top-10 confusions: Gemini (definitions)

| True | Predicted | Count |
|---|---|---|
| create_account | registration_problems | 8 |
| cancel_order | check_cancellation_fee | 6 |
| set_up_shipping_address | change_shipping_address | 5 |
| track_order | delivery_period | 5 |
| get_invoice | check_invoice | 4 |
| create_account | switch_account | 3 |
| get_refund | check_refund_policy | 3 |
| change_shipping_address | edit_account | 2 |
| contact_customer_service | contact_human_agent | 2 |
| check_refund_policy | track_refund | 1 |

## Setup and caveats

- Gemini ran with its default reasoning ON: OpenRouter rejects disabling reasoning for this model. Gemma has no reasoning mode. The comparison is therefore not matched on inference-time compute.
- Gemma sees the full label list in its prompt, just like Gemini; tuning teaches the format and label boundaries, not the label set.
- The Gemini "definitions" variant uses label definitions written by us from the label names only (not from inspecting data or errors).
- The data is synthetic, templated Bitext utterances; high scores here may not transfer to real customer messages.
- The test set has 818 rows, so one row is about 0.12 percentage points.
- Tuning hyperparameters (lr 5e-5, checkpoint at step 500 of 750) were chosen on a 200-row validation slice where the tuned model hit 1.000 (a ceiling), so that slice could not discriminate between checkpoints. The tuned model was run on the test set once.
- Training used only 1,000 examples (about 37 per label).
- Single seed, single run per condition: no estimate of run-to-run variance. The McNemar test covers test-row sampling only, not training or sampling randomness.
