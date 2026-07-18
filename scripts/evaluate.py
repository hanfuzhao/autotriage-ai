"""Metrics and plots shared by the training pipeline and the experiments; macro-F1 is the headline metric since the classes are imbalanced."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "data" / "outputs" / "plots"


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Return the full metric bundle for one model."""
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": report[lbl]["precision"],
            "recall": report[lbl]["recall"],
            "f1": report[lbl]["f1-score"],
            "support": int(report[lbl]["support"]),
        }
        for lbl in labels
        if lbl in report
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
    }


def _short(labels: list[str]) -> list[str]:
    """Compact class names for plot axes."""
    return [l.replace(" SYSTEM", "").replace("DRIVER ASSISTANCE (ADAS)", "ADAS")[:16] for l in labels]


def plot_confusion(y_true, y_pred, labels: list[str], title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = _short(labels)
    ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_comparison(results: dict, path: Path) -> None:
    """Grouped bar chart: accuracy / macro-F1 / weighted-F1 per model."""
    models = list(results.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = [results[m][k] for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x + width * (len(models) - 1) / 2,
                  ["Accuracy", "Macro-F1", "Weighted-F1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_class_distribution(counts: dict, path: Path) -> None:
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names = _short([k for k, _ in items])
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, vals, color="#3b6fb0")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Complaints")
    ax.set_title("Component class distribution (long-tailed)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning_curve(sizes, curves: dict, path: Path, ylabel: str = "Macro-F1") -> None:
    """curves: {model_name: [scores aligned with sizes]}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, scores in curves.items():
        ax.plot(sizes, scores, marker="o", label=name)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(ylabel)
    ax.set_title("Training-set-size sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(levels, curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, scores in curves.items():
        ax.plot(levels, scores, marker="s", label=name)
    ax.set_xlabel("Character-noise rate")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Robustness to noisy / typo-laden complaints")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence_coverage(rows: list[dict], path: Path) -> None:
    """rows: [{threshold, coverage, accuracy}]."""
    cov = [r["coverage"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cov, acc, marker="o", color="#b03b6f")
    for r in rows:
        ax.annotate(f"t={r['threshold']:.1f}", (r["coverage"], r["accuracy"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Accuracy on answered")
    ax.set_title("Confidence-gated abstention (deployment triage)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
