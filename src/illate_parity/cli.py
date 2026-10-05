"""illate-parity command line. Run `illate-parity <command> -h` for options."""

from __future__ import annotations

import argparse
import sys


def cmd_eval(a) -> int:
    from illate_parity.report import evaluate_files

    s = evaluate_files(a.gold, a.pred, a.out, a.labels, a.name)
    lo, hi = s["accuracy_ci"]
    print(
        f"{s['name']}: accuracy {100 * s['accuracy']:.1f}% (95% CI {100 * lo:.1f}-"
        f"{100 * hi:.1f}), macro-F1 {100 * s['macro_f1']:.1f}%, invalid {s['invalid']}"
        f" -> {a.out}/report.md"
    )
    return 0


def cmd_check(a) -> int:
    from illate_parity.parity import run_files

    r = run_files(
        a.gold, a.current, a.candidate, a.out, a.margin, a.labels, a.name_current, a.name_candidate
    )
    print(f"Verdict: {r['verdict']} -> {a.out}/parity.md")
    return 0 if r["verdict"] == "PASS" else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="illate-parity")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("eval", help="score one model's predictions against gold labels")
    s.add_argument("--gold", required=True, help="gold JSONL: id, text, label")
    s.add_argument("--pred", required=True, help="predictions JSONL: id, label")
    s.add_argument("--out", required=True, help="folder for report.md and summary.json")
    s.add_argument("--labels", help="labels.json (keeps classes nobody predicted in the report)")
    s.add_argument("--name", help="model name for the report")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("check", help="is the candidate as good as the current model?")
    s.add_argument("--gold", help="gold labels; omit to score by agreement with current")
    s.add_argument("--current", required=True, help="current model predictions JSONL")
    s.add_argument("--candidate", required=True, help="candidate model predictions JSONL")
    s.add_argument("--out", required=True, help="folder for parity.md and parity.json")
    s.add_argument("--margin", type=float, default=0.03, help="0.03 = 3 accuracy points")
    s.add_argument("--labels", help="labels.json")
    s.add_argument("--name-current", default="current model")
    s.add_argument("--name-candidate", default="candidate")
    s.set_defaults(fn=cmd_check)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
