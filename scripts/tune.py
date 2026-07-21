"""Hyperparameter search for the classical and deep models.

Selection is done on the validation split only. The test split is never touched
here, so the numbers reported in the results section stay an honest held-out
estimate. Scope is deliberately small: a grid over the few knobs that actually
move macro-F1, rather than an exhaustive sweep we could not justify.

    python -m scripts.tune            # run both searches
    python -m scripts.tune --quick    # smaller grids, for a fast check
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from scripts.data import get_class_names, load_splits
from scripts.model import ClassicalModel, TextCNNModel

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "outputs" / "tuning.json"
PLOT_PATH = BASE_DIR / "data" / "outputs" / "plots" / "hyperparameter_tuning.png"


def _macro_f1(model, texts, labels, class_names) -> float:
    return float(f1_score(labels, model.predict(texts), labels=class_names,
                          average="macro", zero_division=0))


def tune_classical(train, val, class_names, quick: bool = False) -> dict:
    """Grid search the TF-IDF and logistic regression knobs on validation."""
    c_values = [1.0, 3.0] if quick else [0.5, 1.0, 3.0, 10.0]
    ngram_values = [2] if quick else [1, 2]

    trials = []
    for c in c_values:
        for ngram_max in ngram_values:
            t0 = time.time()
            model = ClassicalModel(C=c, ngram_max=ngram_max)
            model.fit(train["text"], train["label"])
            score = _macro_f1(model, val["text"], val["label"], class_names)
            trials.append({
                "C": c,
                "ngram_max": ngram_max,
                "val_macro_f1": round(score, 4),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"    C={c:<5} ngram_max={ngram_max}  val_macro_f1={score:.4f}", flush=True)

    best = max(trials, key=lambda t: t["val_macro_f1"])
    return {"search_space": {"C": c_values, "ngram_max": ngram_values},
            "trials": trials, "best": best}


def tune_deep(train, val, class_names, quick: bool = False) -> dict:
    """Small sweep over the TextCNN knobs that matter most, on validation."""
    dropout_values = [0.4] if quick else [0.3, 0.4, 0.5]
    filter_values = [160] if quick else [128, 160]
    epochs = 4 if quick else 10

    trials = []
    for dropout in dropout_values:
        for num_filters in filter_values:
            t0 = time.time()
            model = TextCNNModel(dropout=dropout, num_filters=num_filters,
                                 max_epochs=epochs)
            model.fit(train["text"], train["label"], val["text"], val["label"],
                      verbose=False)
            score = _macro_f1(model, val["text"], val["label"], class_names)
            trials.append({
                "dropout": dropout,
                "num_filters": num_filters,
                "val_macro_f1": round(score, 4),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"    dropout={dropout} filters={num_filters}  "
                  f"val_macro_f1={score:.4f}", flush=True)

    best = max(trials, key=lambda t: t["val_macro_f1"])
    return {"search_space": {"dropout": dropout_values, "num_filters": filter_values,
                             "max_epochs": epochs},
            "trials": trials, "best": best}


def plot_tuning(results: dict, path: Path = PLOT_PATH) -> None:
    """Plot validation macro-F1 for every configuration tried."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cl = results["classical"]["trials"]
    labels = [f"C={t['C']}\nngram<={t['ngram_max']}" for t in cl]
    scores = [t["val_macro_f1"] for t in cl]
    best_i = scores.index(max(scores))
    axes[0].bar(range(len(cl)), scores,
                color=["#c2410c" if i == best_i else "#94a3b8" for i in range(len(cl))])
    axes[0].set_xticks(range(len(cl)), labels, fontsize=8)
    axes[0].set_ylim(min(scores) - 0.02, max(scores) + 0.01)
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("Classical: TF-IDF and LogReg grid")

    dp = results["deep"]["trials"]
    labels = [f"do={t['dropout']}\nfilters={t['num_filters']}" for t in dp]
    scores = [t["val_macro_f1"] for t in dp]
    best_i = scores.index(max(scores))
    axes[1].bar(range(len(dp)), scores,
                color=["#c2410c" if i == best_i else "#94a3b8" for i in range(len(dp))])
    axes[1].set_xticks(range(len(dp)), labels, fontsize=8)
    axes[1].set_ylim(min(scores) - 0.02, max(scores) + 0.01)
    axes[1].set_title("Deep: TextCNN sweep")

    fig.suptitle("Hyperparameter search, selected on the validation split", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(quick: bool = False) -> dict:
    """Run both searches and write tuning.json plus the plot."""
    train, val, _ = load_splits()
    class_names = get_class_names()

    print("Tuning classical model...")
    classical = tune_classical(train, val, class_names, quick)
    print(f"  best: {classical['best']}")

    print("Tuning deep model...")
    deep = tune_deep(train, val, class_names, quick)
    print(f"  best: {deep['best']}")

    results = {
        "protocol": (
            "Grid search. Every configuration is fit on the training split and "
            "scored by macro-F1 on the validation split. The test split is not "
            "used for selection. Seed is fixed at 42 throughout."
        ),
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "classical": classical,
        "deep": deep,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    plot_tuning(results)
    print(f"\nWrote {OUT_PATH.name} and {PLOT_PATH.name}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--quick", action="store_true", help="smaller grids")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
