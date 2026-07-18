"""Experiment harness for the NHTSA component classifier.

Four probes, all reusable and each runnable in isolation:

  1. training_size_experiment  - learning curves: how much labelled data does
       each model need? (the "cold-start" question for a newly tracked model)
  2. robustness_experiment     - degradation under character-level noise, since
       real owner complaints are full of typos and inconsistent casing.
  3. confidence_gating         - accuracy vs. coverage when the deployed model is
       allowed to abstain below a confidence threshold (deployment triage).
  4. head_tail_analysis        - per-class F1 split into frequent (head) and
       rare (tail) safety components.

The primary, written-up experiment is #1 (training-set-size sensitivity); the
others are supporting analyses that inform deployment.
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from scripts.model import ClassicalModel, NaiveBaseline, TextCNNModel

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# 1. Training-set-size sensitivity (primary experiment)
# ---------------------------------------------------------------------------
def training_size_experiment(
    train, val, test, labels, fractions=(0.05, 0.1, 0.25, 0.5, 1.0),
    include_deep: bool = True, deep_epochs: int = 12, seed: int = 42,
) -> dict:
    """Train each model on growing subsets and score macro-F1 on the test set."""
    rng = np.random.default_rng(seed)
    sizes, naive_scores, classical_scores, deep_scores = [], [], [], []
    y_test = test["label"].tolist()

    for frac in fractions:
        n = max(len(labels) * 5, int(len(train) * frac))
        idx = rng.choice(len(train), size=min(n, len(train)), replace=False)
        sub = train.iloc[idx]
        sizes.append(int(len(sub)))

        naive = NaiveBaseline().fit(sub["text"], sub["label"])
        naive_scores.append(f1_score(y_test, naive.predict(test["text"]), labels=labels,
                                     average="macro", zero_division=0))

        clf = ClassicalModel().fit(sub["text"], sub["label"])
        classical_scores.append(f1_score(y_test, clf.predict(test["text"]), labels=labels,
                                         average="macro", zero_division=0))

        if include_deep:
            deep = TextCNNModel(max_epochs=deep_epochs, seed=seed)
            deep.fit(sub["text"], sub["label"], val["text"], val["label"], verbose=False)
            deep_scores.append(f1_score(y_test, deep.predict(test["text"]), labels=labels,
                                        average="macro", zero_division=0))

    curves = {"Naive": naive_scores, "Classical (TF-IDF+LogReg)": classical_scores}
    if include_deep:
        curves["Deep (TextCNN)"] = deep_scores
    return {"sizes": sizes, "fractions": list(fractions), "curves": curves}


# ---------------------------------------------------------------------------
# 2. Robustness to character noise
# ---------------------------------------------------------------------------
def add_char_noise(text: str, rate: float, rng: random.Random) -> str:
    """Corrupt a fraction of characters with typo-like edits."""
    if rate <= 0:
        return text
    out = []
    for ch in text:
        if ch != " " and rng.random() < rate:
            op = rng.random()
            if op < 0.34:
                continue  # deletion
            elif op < 0.67:
                out.append(rng.choice(_ALPHABET))  # substitution
            else:
                out.append(ch); out.append(rng.choice(_ALPHABET))  # insertion
        else:
            out.append(ch)
    return "".join(out)


def robustness_experiment(models: dict, test, labels, levels=(0.0, 0.05, 0.1, 0.2, 0.3), seed: int = 42) -> dict:
    """Score each fitted model on increasingly noisy versions of the test set."""
    curves: dict[str, list[float]] = {name: [] for name in models}
    y_test = test["label"].tolist()
    for lvl in levels:
        rng = random.Random(seed)
        noisy = [add_char_noise(t, lvl, rng) for t in test["text"].tolist()]
        for name, model in models.items():
            pred = model.predict(noisy)
            curves[name].append(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0))
    return {"levels": list(levels), "curves": curves}


# ---------------------------------------------------------------------------
# 3. Confidence-gated abstention
# ---------------------------------------------------------------------------
def confidence_gating(model, test, labels, thresholds=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    """Accuracy and coverage when the model answers only above each threshold."""
    proba = model.predict_proba(test["text"])
    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    pred = np.array([model.classes_[i] for i in pred_idx])
    y_true = np.array(test["label"].tolist())
    rows = []
    for t in thresholds:
        mask = conf >= t
        cov = float(mask.mean())
        acc = float(accuracy_score(y_true[mask], pred[mask])) if mask.any() else 0.0
        rows.append({"threshold": float(t), "coverage": cov, "accuracy": acc})
    return rows


# ---------------------------------------------------------------------------
# 4. Head vs tail per-class analysis
# ---------------------------------------------------------------------------
def head_tail_analysis(per_class: dict, class_counts: dict, head_k: int = 5) -> dict:
    """Split classes into head (most frequent) and tail; average their F1."""
    ordered = sorted(class_counts, key=lambda c: class_counts[c], reverse=True)
    head, tail = ordered[:head_k], ordered[head_k:]
    mean_f1 = lambda group: float(np.mean([per_class[c]["f1"] for c in group if c in per_class])) if group else 0.0
    return {
        "head_classes": head, "tail_classes": tail,
        "head_macro_f1": mean_f1(head), "tail_macro_f1": mean_f1(tail),
    }


# ---------------------------------------------------------------------------
# Error analysis: surface confident mistakes for the write-up
# ---------------------------------------------------------------------------
def collect_errors(model, test, n: int = 12) -> list[dict]:
    """Return the most confident misclassifications for manual error analysis."""
    proba = model.predict_proba(test["text"])
    conf = proba.max(axis=1)
    pred = np.array([model.classes_[i] for i in proba.argmax(axis=1)])
    y_true = np.array(test["label"].tolist())
    texts = test["text"].tolist()
    wrong = np.where(pred != y_true)[0]
    wrong = sorted(wrong, key=lambda i: conf[i], reverse=True)[:n]
    return [
        {
            "text": texts[i][:400],
            "true": str(y_true[i]),
            "pred": str(pred[i]),
            "confidence": float(conf[i]),
        }
        for i in wrong
    ]
