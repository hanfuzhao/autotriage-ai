"""Exploratory data analysis for the NHTSA complaints corpus."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.data import get_class_names
from scripts.evaluate import plot_class_distribution

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "outputs"


def summarize(df) -> dict:
    """Compute dataset-level EDA statistics."""
    lengths = df["n_words"].to_numpy() if "n_words" in df else df["text"].str.split().str.len().to_numpy()
    counts = df["label"].value_counts().to_dict()
    stats = {
        "n_examples": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio": float(max(counts.values()) / max(1, min(counts.values()))),
        "words_mean": float(np.mean(lengths)),
        "words_median": float(np.median(lengths)),
        "words_p90": float(np.percentile(lengths, 90)),
    }
    for flag in ("crash", "fire"):
        if flag in df:
            stats[f"{flag}_rate"] = float(df[flag].mean())
    if "numberOfInjuries" in df:
        stats["injury_rate"] = float((df["numberOfInjuries"] > 0).mean())
    return stats


def run_eda(df, save_plot: bool = True) -> dict:
    stats = summarize(df)
    if save_plot:
        plot_class_distribution(stats["class_counts"], OUT_DIR / "plots" / "class_distribution.png")
    return stats


if __name__ == "__main__":
    from scripts.data import load_splits
    import pandas as pd

    train, val, test = load_splits()
    stats = run_eda(pd.concat([train, val, test], ignore_index=True))
    print(json.dumps(stats, indent=2))
