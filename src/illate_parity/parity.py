"""Parity check: is candidate model B as good as the client's current model A?

This is a non-inferiority test, not "is B better". The client agrees a margin up
front (default 3 accuracy points on ~1,000 items). Then:

  PASS          the whole 95% CI of (B - A) sits above -margin
  FAIL          the whole CI sits below -margin
  INCONCLUSIVE  the CI straddles -margin: get more labelled items, don't guess

With no gold labels, B is scored by agreement with A instead. That only shows
B behaves like A, including A's mistakes, and the report says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from illate_parity.bootstrap import bootstrap_ci, fast_paired_accuracy
from illate_parity.io import align, read_jsonl
from illate_parity.metrics import encode, evaluate

MIN_ITEMS = 1000


def _latency(preds: list[dict]) -> dict | None:
    vals = [float(p["latency_ms"]) for p in preds if p.get("latency_ms") is not None]
    if not vals:
        return None
    a = np.array(vals)
    return {
        "n": len(a),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
    }


def _tokens(preds: list[dict]) -> dict | None:
    ins = [p.get("input_tokens") for p in preds]
    outs = [p.get("output_tokens") for p in preds]
    if any(v is None for v in ins + outs):
        return None
    return {"avg_input": float(np.mean(ins)), "avg_output": float(np.mean(outs))}


def verdict(low: float, high: float, margin: float) -> str:
    if low >= -margin:
        return "PASS"
    if high < -margin:
        return "FAIL"
    return "INCONCLUSIVE"


def check(
    gold: list[dict] | None,
    pred_a: list[dict],
    pred_b: list[dict],
    margin: float = 0.03,
    n_boot: int = 10_000,
    seed: int = 0,
    labels: list[str] | None = None,
) -> dict:
    if gold is None:
        # agreement mode: A's answers are the reference
        ref = [{"id": p["id"], "label": p["label"]} for p in pred_a]
        b = align(ref, pred_b, "candidate predictions")
        y, pb, _ = encode([r["label"] for r in ref], [x["label"] for x in b], labels)
        agree, lo, hi = bootstrap_ci(y, pb, n_boot=n_boot, seed=seed)
        return {
            "mode": "agreement",
            "n": len(ref),
            "margin": margin,
            "agreement": {"value": agree, "low": lo, "high": hi},
            "verdict": verdict(lo - 1.0, hi - 1.0, margin),  # B vs A where A scores 1.0
            "latency": {"a": _latency(pred_a), "b": _latency(b)},
            "tokens": {"a": _tokens(pred_a), "b": _tokens(b)},
            "warnings": _warnings(len(ref))
            + ["No gold labels: B is judged by agreeing with A, mistakes included."],
        }

    a = align(gold, pred_a, "current-model predictions")
    b = align(gold, pred_b, "candidate predictions")
    gold_labels = [g["label"] for g in gold]
    if labels is None:
        labels = sorted(set(gold_labels))
    ev_a = evaluate(gold_labels, [x["label"] for x in a], labels)
    ev_b = evaluate(gold_labels, [x["label"] for x in b], labels)
    y, pa, _ = encode(gold_labels, [x["label"] for x in a], labels)
    _, pb, _ = encode(gold_labels, [x["label"] for x in b], labels)

    _, a_lo, a_hi = bootstrap_ci(y, pa, n_boot=n_boot, seed=seed)
    _, b_lo, b_hi = bootstrap_ci(y, pb, n_boot=n_boot, seed=seed + 1)
    diff = fast_paired_accuracy(y, pa, pb, n_boot=n_boot, seed=seed + 2)

    # classes where B lost the most F1 against A: what to show the client first
    a_f1 = {s.label: s.f1 for s in ev_a["per_class"]}
    drops = sorted(
        ((s.label, a_f1[s.label], s.f1, s.support) for s in ev_b["per_class"] if s.support),
        key=lambda t: t[2] - t[1],
    )[:10]

    return {
        "mode": "gold",
        "n": len(gold),
        "margin": margin,
        "a": {
            "accuracy": ev_a["accuracy"],
            "low": a_lo,
            "high": a_hi,
            "macro_f1": ev_a["macro_f1"],
            "invalid": ev_a["invalid"],
        },
        "b": {
            "accuracy": ev_b["accuracy"],
            "low": b_lo,
            "high": b_hi,
            "macro_f1": ev_b["macro_f1"],
            "invalid": ev_b["invalid"],
        },
        "diff": diff,
        "verdict": verdict(diff["low"], diff["high"], margin),
        "worst_classes_for_b": [
            {"label": lab, "f1_a": fa, "f1_b": fb, "support": n} for lab, fa, fb, n in drops
        ],
        "b_top_confusions": ev_b["top_confusions"],
        "latency": {"a": _latency(a), "b": _latency(b)},
        "tokens": {"a": _tokens(a), "b": _tokens(b)},
        "warnings": _warnings(len(gold)),
    }


def _warnings(n: int) -> list[str]:
    if n < MIN_ITEMS:
        return [f"Only {n} items. Use {MIN_ITEMS}+ for a 3-point margin; small sets give wide CIs."]
    return []


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(r: dict, name_a: str = "current model", name_b: str = "candidate") -> str:
    m = _pct(r["margin"])
    out = [
        f"# Parity check: {name_b} vs {name_a}",
        "",
        f"**Verdict: {r['verdict']}** (non-inferiority margin {m}, "
        f"{r['n']} items, 95% CIs by bootstrap)",
        "",
    ]
    if r["mode"] == "agreement":
        ag = r["agreement"]
        out += [
            f"- {name_b} agrees with {name_a} on {_pct(ag['value'])} of items "
            f"(CI {_pct(ag['low'])} to {_pct(ag['high'])}).",
            "",
        ]
    else:
        a, b, d = r["a"], r["b"], r["diff"]
        out += [
            "| Model | Accuracy | 95% CI | Macro-F1 | Invalid outputs |",
            "|---|---:|---|---:|---:|",
            f"| {name_a} | {_pct(a['accuracy'])} | {_pct(a['low'])} to {_pct(a['high'])} "
            f"| {_pct(a['macro_f1'])} | {a['invalid']} |",
            f"| {name_b} | {_pct(b['accuracy'])} | {_pct(b['low'])} to {_pct(b['high'])} "
            f"| {_pct(b['macro_f1'])} | {b['invalid']} |",
            "",
            f"Difference ({name_b} minus {name_a}): **{100 * d['diff']:+.1f} points**, "
            f"paired 95% CI {100 * d['low']:+.1f} to {100 * d['high']:+.1f}. "
            f"{name_b} is worse in {_pct(d['p_b_worse'])} of resamples.",
            "",
            f"### Classes where {name_b} loses most (F1)",
            "| Class | F1 current | F1 candidate | Items |",
            "|---|---:|---:|---:|",
        ]
        out += [
            f"| {c['label']} | {_pct(c['f1_a'])} | {_pct(c['f1_b'])} | {c['support']} |"
            for c in r["worst_classes_for_b"][:5]
        ]
        out += [
            "",
            f"### {name_b}'s most common mistakes",
            "| True | Predicted | Count |",
            "|---|---|---:|",
        ]
        out += [f"| {t} | {p} | {c} |" for t, p, c in r["b_top_confusions"][:5]]
        out.append("")
    lat = r["latency"]
    if lat["a"] or lat["b"]:
        out += ["### Latency", "| Model | p50 ms | p95 ms |", "|---|---:|---:|"]
        for name, key in ((name_a, "a"), (name_b, "b")):
            if lat[key]:
                out.append(f"| {name} | {lat[key]['p50_ms']:.0f} | {lat[key]['p95_ms']:.0f} |")
        out.append("")
    if r["warnings"]:
        out += ["### Warnings", *[f"- {w}" for w in r["warnings"]], ""]
    return "\n".join(out)


def run_files(
    gold_path,
    a_path,
    b_path,
    out_dir,
    margin=0.03,
    labels_path=None,
    name_a="current model",
    name_b="candidate",
) -> dict:
    gold = read_jsonl(gold_path) if gold_path else None
    labels = json.loads(Path(labels_path).read_text()) if labels_path else None
    r = check(gold, read_jsonl(a_path), read_jsonl(b_path), margin=margin, labels=labels)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps(r, indent=2, default=float) + "\n")
    (out / "parity.md").write_text(render(r, name_a, name_b))
    return r
