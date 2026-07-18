"""Training pipeline. Downloads or reuses the data, builds the splits, trains all
three models, evaluates on the test set, runs the experiments, and writes
metrics.json and the plots. Trained models go in models/ for the app to serve.

    python setup.py                 # full run
    python setup.py --fast          # tiny run to check the plumbing
    python setup.py --skip-download # reuse an existing data/raw dump
    python setup.py --no-deep       # skip the neural model
"""
# Parts of this project were drafted with an AI assistant (Claude) and then reviewed and edited.
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from scripts import data as data_mod
from scripts import evaluate as ev
from scripts import experiment as exp
from scripts.eda import run_eda
from scripts.model import ClassicalModel, NaiveBaseline, TextCNNModel

MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "data" / "outputs"
PLOTS_DIR = OUT_DIR / "plots"


def _ensure_raw(skip_download: bool) -> None:
    """Make sure a raw corpus exists; download it or fall back to the sample."""
    if data_mod.RAW_PATH.exists():
        return
    if skip_download and data_mod.SAMPLE_PATH.exists():
        print("Raw dump missing; falling back to committed sample.csv")
        sample = pd.read_csv(data_mod.SAMPLE_PATH)
        data_mod.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        # reconstruct a minimal raw jsonl from the sample so the pipeline runs
        with data_mod.RAW_PATH.open("w", encoding="utf-8") as fh:
            for _, r in sample.iterrows():
                fh.write(json.dumps({
                    "odiNumber": None, "make": r.get("make"), "model": r.get("model"),
                    "modelYear": r.get("modelYear"), "summary": r["text"],
                    "components": r["label"], "crash": bool(r.get("crash", False)),
                    "fire": bool(r.get("fire", False)),
                    "numberOfInjuries": int(r.get("numberOfInjuries", 0) or 0),
                    "numberOfDeaths": 0, "dateComplaintFiled": None,
                }) + "\n")
        return
    print("Downloading NHTSA corpus (this can take a few minutes)...")
    from scripts.make_dataset import MAKES, YEARS, download
    download(MAKES, YEARS, out_path=data_mod.RAW_PATH)


def _subsample_for_fast(train, val, test):
    train = train.groupby("label", group_keys=False).head(80)
    val = val.groupby("label", group_keys=False).head(20)
    test = test.groupby("label", group_keys=False).head(25)
    return train, val, test


def train_and_evaluate(fast: bool, no_deep: bool, skip_download: bool) -> dict:
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_raw(skip_download)
    stats = data_mod.build_and_save()
    train, val, test = data_mod.load_splits()
    if fast:
        train, val, test = _subsample_for_fast(train, val, test)
    labels = data_mod.get_class_names()
    labels = [l for l in labels if l in set(train["label"])]  # drop empty classes in fast mode
    print(f"Data: train={len(train)} val={len(val)} test={len(test)} classes={len(labels)}")

    eda_stats = run_eda(pd.concat([train, val, test], ignore_index=True))

    print("\n[1/3] Naive baseline...")
    naive = NaiveBaseline().fit(train["text"], train["label"])
    naive.save(MODELS_DIR / "naive.pkl")

    print("[2/3] Classical TF-IDF + LogReg...")
    classical = ClassicalModel().fit(train["text"], train["label"])
    classical.save(MODELS_DIR / "classical.pkl")

    results = {}
    for name, model in (("naive", naive), ("classical", classical)):
        pred = model.predict(test["text"])
        results[name] = ev.compute_metrics(test["label"], pred, labels)
        ev.plot_confusion(test["label"], pred, labels, f"{name} - confusion (row-normalised)",
                          PLOTS_DIR / f"confusion_{name}.png")

    deep = None
    if not no_deep:
        print("[3/3] Deep TextCNN...")
        epochs = 3 if fast else 25
        deep = TextCNNModel(max_epochs=epochs).fit(
            train["text"], train["label"], val["text"], val["label"], verbose=True
        )
        deep.save(MODELS_DIR / "deep_textcnn.pt")
        pred = deep.predict(test["text"])
        results["deep"] = ev.compute_metrics(test["label"], pred, labels)
        ev.plot_confusion(test["label"], pred, labels, "deep (TextCNN) - confusion (row-normalised)",
                          PLOTS_DIR / "confusion_deep.png")

    ev.plot_model_comparison(results, PLOTS_DIR / "model_comparison.png")

    (MODELS_DIR / "labels.json").write_text(json.dumps({
        "classes": labels, "deployed_model": "deep" if deep is not None else "classical",
    }, indent=2))

    experiments: dict = {}
    errors: dict = {}
    deployed = deep if deep is not None else classical
    fitted = {"Classical (TF-IDF+LogReg)": classical}
    if deep is not None:
        fitted["Deep (TextCNN)"] = deep

    if not fast:
        print("\n[exp] training-size sensitivity...")
        ts = exp.training_size_experiment(train, val, test, labels, include_deep=not no_deep)
        experiments["training_size"] = ts
        ev.plot_learning_curve(ts["sizes"], ts["curves"], PLOTS_DIR / "learning_curve.png")

        print("[exp] robustness to noise...")
        rob = exp.robustness_experiment(fitted, test, labels)
        experiments["robustness"] = rob
        ev.plot_robustness(rob["levels"], rob["curves"], PLOTS_DIR / "robustness.png")

        print("[exp] confidence gating...")
        gate = exp.confidence_gating(deployed, test, labels)
        experiments["confidence_gating"] = gate
        ev.plot_confidence_coverage(gate, PLOTS_DIR / "confidence_coverage.png")

        print("[exp] head/tail analysis + error mining...")
        experiments["head_tail"] = {
            "classical": exp.head_tail_analysis(results["classical"]["per_class"], stats["class_counts"]),
        }
        if deep is not None:
            experiments["head_tail"]["deep"] = exp.head_tail_analysis(
                results["deep"]["per_class"], stats["class_counts"])
        errors["deployed"] = exp.collect_errors(deployed, test, n=12)

    metrics = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": data_mod.RANDOM_SEED,
            "fast_mode": fast,
            "runtime_seconds": round(time.time() - t0, 1),
            "dataset": stats,
            "eda": eda_stats,
            "labels": labels,
        },
        "models": results,
        "experiments": experiments,
        "errors": errors,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone in {metrics['meta']['runtime_seconds']}s. Wrote metrics.json + plots.")
    _print_summary(results)
    return metrics


def _print_summary(results: dict) -> None:
    print("\n=== Test-set summary ===")
    print(f"{'model':<12}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>13}")
    for name, m in results.items():
        print(f"{name:<12}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}{m['weighted_f1']:>13.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NHTSA component classifier")
    parser.add_argument("--fast", action="store_true", help="tiny smoke-test run")
    parser.add_argument("--no-deep", action="store_true", help="skip the neural model")
    parser.add_argument("--skip-download", action="store_true", help="reuse existing raw data")
    args = parser.parse_args()
    train_and_evaluate(fast=args.fast, no_deep=args.no_deep, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
