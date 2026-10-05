"""Classification metrics, written out by hand so every number can be explained."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def encode(gold: list[str], pred: list[str], labels: list[str] | None = None):
    """Map string labels to ints. Predictions outside the label set (a model that
    answered 'I am not sure') get their own index len(labels), which always counts
    as wrong."""
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} items, pred has {len(pred)}")
    if labels is None:
        labels = sorted(set(gold))
    index = {lab: i for i, lab in enumerate(labels)}
    invalid = len(labels)
    y = np.array([index[g] for g in gold], dtype=np.int64)
    p = np.array([index.get(x, invalid) for x in pred], dtype=np.int64)
    return y, p, labels


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(y == p)) if len(y) else float("nan")


def confusion_matrix(y: np.ndarray, p: np.ndarray, n_labels: int) -> np.ndarray:
    """Rows = true class, columns = predicted class. The extra last column counts
    invalid predictions (outside the label set)."""
    cm = np.zeros((n_labels, n_labels + 1), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


@dataclass
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


def per_class(cm: np.ndarray, labels: list[str]) -> list[ClassScore]:
    tp = np.diag(cm[:, : len(labels)])
    predicted = cm[:, : len(labels)].sum(axis=0)  # how often each class was predicted
    support = cm.sum(axis=1)  # how often each class is the truth
    out = []
    for i, lab in enumerate(labels):
        prec = tp[i] / predicted[i] if predicted[i] else 0.0
        rec = tp[i] / support[i] if support[i] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append(ClassScore(lab, float(prec), float(rec), float(f1), int(support[i])))
    return out


def macro_f1(scores: list[ClassScore]) -> float:
    present = [s.f1 for s in scores if s.support > 0]
    return float(np.mean(present)) if present else float("nan")


def top_confusions(cm: np.ndarray, labels: list[str], k: int = 10) -> list[tuple[str, str, int]]:
    """The k most frequent (true, predicted) mistakes."""
    names = labels + ["<invalid>"]
    off = cm.copy()
    for i in range(len(labels)):
        off[i, i] = 0
    flat = np.argsort(off, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        out.append((names[i], names[j], int(off[i, j])))
    return out


def evaluate(gold: list[str], pred: list[str], labels: list[str] | None = None) -> dict:
    y, p, labels = encode(gold, pred, labels)
    cm = confusion_matrix(y, p, len(labels))
    scores = per_class(cm, labels)
    return {
        "n": int(len(y)),
        "accuracy": accuracy(y, p),
        "macro_f1": macro_f1(scores),
        "invalid": int(cm[:, -1].sum()),
        "per_class": scores,
        "confusion": cm,
        "labels": labels,
        "top_confusions": top_confusions(cm, labels),
    }
