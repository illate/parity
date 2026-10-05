# illate-parity

Is a cheaper model as good as the one you run now?

`illate-parity` answers that from two prediction files. It scores each model with a
bootstrap confidence interval, then runs a **non-inferiority test**: does the new model
stay within an agreed margin of the current one, with 95% confidence?

It is the check [ILLATE](https://illate.dev) runs before moving a team off a fine-tuned
API model. No GPU, no API keys, one dependency (NumPy).

## Install

```bash
pip install git+https://github.com/illate/parity
```

## Use

You need a file of gold labels and one predictions file per model, all JSONL:

```jsonl
{"id": "1", "text": "I still haven't got my card", "label": "card_arrival"}   gold
{"id": "1", "label": "card_arrival", "latency_ms": 41.2}                       predictions
```

`latency_ms` is optional. Items match on `id`.

```bash
# accuracy, macro-F1, per-class scores and top confusions for one model
illate-parity eval --gold gold.jsonl --pred current.jsonl --out out/current

# is the candidate within 3 accuracy points of the current model?
illate-parity check --gold gold.jsonl --current current.jsonl --candidate candidate.jsonl \
  --margin 0.03 --out out/parity
```

`check` writes `parity.md` and `parity.json` and exits 0 only on PASS, so it can gate CI.

### Verdicts

| Verdict | Meaning |
|---|---|
| PASS | The whole 95% CI of (candidate − current) is above −margin. |
| FAIL | The whole CI is below −margin. |
| INCONCLUSIVE | The CI straddles −margin. Label more items; don't guess. |

With no `--gold`, the candidate is scored by agreement with the current model. That only
shows it behaves like the current model, mistakes included, and the report says so.
Below 1,000 items the report warns that intervals will be wide.

## Example: Banking77

`examples/banking77/` holds the Banking77 test set (3,080 messages, 77 intents) and
predictions from two models: a TF-IDF + logistic regression baseline and a fine-tuned
Qwen3-4B LoRA ([weights on Hugging Face](https://huggingface.co/illate/qwen3-4b-banking77-lora)).

```bash
illate-parity check --gold examples/banking77/test.jsonl \
  --current examples/banking77/tfidf.preds.jsonl \
  --candidate examples/banking77/qwen3-4b-lora.preds.jsonl \
  --labels examples/banking77/labels.json \
  --name-current "TF-IDF baseline" --name-candidate "Qwen3-4B LoRA" --out out/example
```

Output (`out/example/parity.md`):

| Model | Accuracy | 95% CI | Macro-F1 | Invalid outputs |
|---|---:|---|---:|---:|
| TF-IDF baseline | 89.3% | 88.1% to 90.3% | 89.3% | 0 |
| Qwen3-4B LoRA | 94.0% | 93.1% to 94.8% | 94.0% | 0 |

**Verdict: PASS.** Difference +4.7 points, paired 95% CI +3.7 to +5.8. The report also
lists the classes where the candidate loses most and its most common mistakes.

## How the statistics work

- Accuracy CIs: 10,000 bootstrap resamples over items.
- The parity test uses a **paired** bootstrap: each resample takes the same items for both
  models, so the interval reflects the difference on identical inputs, not two
  independent noisy scores.
- Labels a model writes that are not in the label set count as wrong and are reported as
  "invalid outputs".

## Data and licences

Code: Apache 2.0. Banking77 data: CC BY 4.0, PolyAI (Casanueva et al., 2020, "Efficient
Intent Detection with Dual Sentence Encoders"), from
[PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets).

Questions or a parity check on your own data: poojith@illate.dev · [illate.dev](https://illate.dev)
