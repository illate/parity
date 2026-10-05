"""Single-model evaluation report (accuracy with CI, macro-F1, per-class, confusions)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from illate_parity.bootstrap import bootstrap_ci
from illate_parity.io import align, read_jsonl
from illate_parity.metrics import encode, evaluate


def evaluate_files(
    gold_path, pred_path, out_dir, labels_path=None, name=None, n_boot=10_000, only_predicted=False
) -> dict:
    gold = read_jsonl(gold_path)
    raw_preds = read_jsonl(pred_path)
    if only_predicted:
        # score a partial run (e.g. a smoke test on the first 200 items) on just those items
        have = {str(r["id"]) for r in raw_preds}
        gold = [g for g in gold if str(g["id"]) in have]
    preds = align(gold, raw_preds)
    labels = json.loads(Path(labels_path).read_text()) if labels_path else None
    g = [r["label"] for r in gold]
    p = [r["label"] for r in preds]
    ev = evaluate(g, p, labels)
    y, yp, _ = encode(g, p, ev["labels"])
    acc, lo, hi = bootstrap_ci(y, yp, n_boot=n_boot)
    lat = [r["latency_ms"] for r in preds if r.get("latency_ms") is not None]
    summary = {
        "name": name or Path(pred_path).stem,
        "n": ev["n"],
        "accuracy": acc,
        "accuracy_ci": [lo, hi],
        "macro_f1": ev["macro_f1"],
        "invalid": ev["invalid"],
        "latency_p50_ms": sorted(lat)[len(lat) // 2] if lat else None,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "precision", "recall", "f1", "support"])
        for s in ev["per_class"]:
            w.writerow([s.label, f"{s.precision:.4f}", f"{s.recall:.4f}", f"{s.f1:.4f}", s.support])
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *ev["labels"], "<invalid>"])
        for lab, row in zip(ev["labels"], ev["confusion"], strict=True):
            w.writerow([lab, *row.tolist()])
    worst = sorted(ev["per_class"], key=lambda s: s.f1)[:10]
    md = [
        f"# Eval: {summary['name']}",
        "",
        f"- Items: {ev['n']}",
        f"- Accuracy: **{100 * acc:.1f}%** (95% CI {100 * lo:.1f} to {100 * hi:.1f})",
        f"- Macro-F1: {100 * ev['macro_f1']:.1f}%",
        f"- Invalid outputs: {ev['invalid']}",
    ]
    if summary["latency_p50_ms"] is not None:
        md.append(f"- Latency p50: {summary['latency_p50_ms']:.0f} ms")
    md += ["", "## Weakest classes", "| Class | F1 | Items |", "|---|---:|---:|"]
    md += [f"| {s.label} | {100 * s.f1:.1f}% | {s.support} |" for s in worst]
    md += ["", "## Most common mistakes", "| True | Predicted | Count |", "|---|---|---:|"]
    md += [f"| {t} | {pr} | {c} |" for t, pr, c in ev["top_confusions"]]
    (out / "report.md").write_text("\n".join(md) + "\n")
    return summary
