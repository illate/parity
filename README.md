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

## Replacing a fine-tuned gpt-4.1-nano: GPT-5.6 Luna vs a model you own

OpenAI is scheduled to stop serving fine-tunes of gpt-4.1-nano, o4-mini, gpt-3.5-turbo and
gpt-4 on 23 October 2026. For fine-tuned gpt-4.1-nano the named replacement is gpt-5.6-luna
([deprecations page](https://developers.openai.com/api/docs/deprecations), checked 8 October 2026),
and Luna can't be fine-tuned ([model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
checked 8 October 2026). That leaves one question: is the replacement as good as what you had,
on your data? This tool answers it with a paired test. It needs no GPU and no API key: you run
the models yourself and give it their prediction files.

### 1. Before 23 October: save your current model's answers

- Pick 2,000 to 5,000 real inputs, chosen so rare classes are covered.
- Label them with ground truth written or checked by people, or taken from your records. Don't
  use your fine-tune's answers as the labels.
- Run your fine-tune on them in your own OpenAI organization, with your own key, at normal rate
  limits. Record the model ID, prompt, parameters, timestamp and raw output.
- Keep logging the answers it gives on live traffic until the shutdown.

If it stops on 23 October as scheduled, nobody can ask your fine-tune anything afterwards, so this
file (`current.preds.jsonl`) is the only way to measure a replacement against what you had.

### 2. Make one prediction file per candidate

Run each candidate on the same items and write one JSON line per item, with the same `id` as
the gold file and the predicted `label`:

- GPT-5.6 Luna with your best prompt, for example the 20 most similar labelled examples
  retrieved for each item.
- An embeddings classifier, for example bge-base-en-v1.5 plus logistic regression.
- A model you own: a fine-tuned open model or a fine-tuned encoder.
- If you qualify: a gpt-4.1-mini fine-tune (no new OpenAI fine-tuning jobs for anyone from
  6 January 2027), or Azure, if your subscription has already deployed gpt-4.1-nano.

### 3. Check each candidate against the current model

    pip install git+https://github.com/illate/parity
    illate-parity check --gold test.jsonl \
      --current current.preds.jsonl --candidate luna.preds.jsonl \
      --margin 0.03 --out out/luna

- `--margin 0.03` means the candidate may be at most 3 accuracy points worse. Agree the margin
  before you look at results.
- PASS means the whole 95% confidence interval of (candidate minus current) sits above minus the
  margin. FAIL means it sits entirely below. INCONCLUSIVE means you need more labelled items.
- The command exits non-zero unless the verdict is PASS.
- Run it once per candidate.

### What this looked like on public data

Measured by ILLATE, 5 to 8 October 2026 (UTC). Each setting ran once on the full held-out test
set. Accuracy is shown with its 95% bootstrap CI. Paired differences are the fine-tuned model minus
the other model, with a paired 95% CI. The Luna runs with retrieved examples were pre-registered on
all three sets.

| Test set | Fine-tuned Qwen3-4B LoRA | Best embeddings + logistic regression | GPT-5.6 Luna | GPT-6 Luna |
|---|---|---|---|---|
| Banking77 (3,080, 77 intents) | 94.0% (93.1 to 94.8) | 93.5%, bge-base (92.7 to 94.4) | 93.8% (92.9 to 94.6), 20 most similar examples | 93.1% (92.1 to 94.0), 20 most similar examples |
| CLINC150 (5,500, 150 intents + out-of-scope) | 95.1% (94.6 to 95.7) | 90.6%, bge-base (89.8 to 91.4) | 94.6% (94.0 to 95.2), 20 most similar examples | 94.4% (93.8 to 95.0), 20 most similar examples |
| CFPB complaints (5,117, 10 products, merged after seeing the errors) | 89.5% (88.7 to 90.3) | 86.4%, bge-small (85.5 to 87.3) | 87.5% (86.6 to 88.4), 20 most similar examples | 86.0% (85.1 to 87.0), category list only (not run with examples: budget) |

- **On Banking77 three options tied:** GPT-5.6 Luna with the 20 most similar training examples
  (paired difference +0.2 points, 95% CI -0.6 to +0.9), the embeddings classifier (+0.5, -0.3 to
  +1.2) and the fine-tuned model. When candidates tie, pick on cost at your volume, latency and
  control, and check per-class results with this tool.
- **Luna's prompt mattered most.** With one fixed example per category, GPT-5.6 Luna scored 87.2% on Banking77;
  with the 20 most similar, 93.8%. Reasoning effort low didn't help.
- **With the same examples, Luna tied on CLINC150 and trailed by 2 points on CFPB.** CLINC150:
  +0.5 (-0.04 to +1.0), a tie; what's left is an out-of-scope trade-off (the fine-tuned model
  caught more out-of-scope requests, 83.9% against 79.1%, and wrongly rejected slightly more real
  ones, 0.4% against 0.2%). CFPB: +2.0 (+1.2 to +2.9), which clears the 2-point line set before
  the run by about two complaints; credit reporting, the biggest product, accounts for more than
  all of it (117 more complaints right against a gap of 104); counted as complaints right, Luna was
  ahead on mortgages (12), credit cards (11), vehicle loans (7), payday and personal loans (6) and
  money transfer (1).
  The CFPB merge of renamed products was chosen after seeing the errors; as filed, TF-IDF tied the
  fine-tuned model. So the cheapest option that passes differs by task: test them all with this
  tool.
- **Not measured:** reasoning effort medium, the Decisions API with choice descriptions (with bare
  labels it scored 76.8% on Banking77), a fine-tuned encoder, or a real fine-tuned gpt-4.1-nano.
- Full method and limits: [illate.dev/luna-vs-fine-tuned-model](https://illate.dev/luna-vs-fine-tuned-model).
  Raw numbers: [bakeoff.json](https://illate.dev/bakeoff.json).

If Luna or an embeddings classifier passes on your data, use it. Parity is maintained by ILLATE,
which also runs this bake-off for teams. The tool is open source under Apache 2.0 either way.

Not sure whether your own fine-tune is on OpenAI's list: [fine-tune shutdown checker](https://illate.dev/fine-tune-shutdown-checker).

## Data and licences

Code: Apache 2.0. Banking77 data: CC BY 4.0, PolyAI (Casanueva et al., 2020, "Efficient
Intent Detection with Dual Sentence Encoders"), from
[PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets).

Questions or a parity check on your own data: poojith@illate.dev · [illate.dev](https://illate.dev)
