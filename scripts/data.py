"""Loads raw NHTSA complaints, derives labels, cleans text, and makes train/val/test splits."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "nhtsa_complaints.jsonl"
PROC_DIR = BASE_DIR / "data" / "processed"
SAMPLE_PATH = BASE_DIR / "data" / "raw" / "sample.csv"

RANDOM_SEED = 42

# we predict the first-listed component, and merge near-duplicate codes into one taxonomy
COMPONENT_MAP: dict[str, str] = {
    "ELECTRICAL SYSTEM": "ELECTRICAL SYSTEM",
    "ENGINE": "ENGINE",
    "ENGINE AND ENGINE COOLING": "ENGINE",
    "POWER TRAIN": "POWER TRAIN",
    "POWER TRAIN:AUTOMATIC TRANSMISSION": "POWER TRAIN",
    "STEERING": "STEERING",
    "SERVICE BRAKES": "SERVICE BRAKES",
    "SERVICE BRAKES, HYDRAULIC": "SERVICE BRAKES",
    "FUEL/PROPULSION SYSTEM": "FUEL/PROPULSION SYSTEM",
    "FUEL SYSTEM": "FUEL/PROPULSION SYSTEM",
    "FUEL SYSTEM, GASOLINE": "FUEL/PROPULSION SYSTEM",
    "GASOLINE": "FUEL/PROPULSION SYSTEM",
    "AIR BAGS": "AIR BAGS",
    "FORWARD COLLISION AVOIDANCE": "DRIVER ASSISTANCE (ADAS)",
    "LANE DEPARTURE": "DRIVER ASSISTANCE (ADAS)",
    "BACK OVER PREVENTION": "DRIVER ASSISTANCE (ADAS)",
    "EXTERIOR LIGHTING": "EXTERIOR LIGHTING",
    "VISIBILITY/WIPER": "VISIBILITY/WIPER",
    "VISIBILITY": "VISIBILITY/WIPER",
    "STRUCTURE": "STRUCTURE",
    "VEHICLE SPEED CONTROL": "VEHICLE SPEED CONTROL",
    "SUSPENSION": "SUSPENSION",
    "SEAT BELTS": "SEATS/SEAT BELTS",
    "SEATS": "SEATS/SEAT BELTS",
}

CANONICAL_LABELS: list[str] = [
    "ELECTRICAL SYSTEM",
    "ENGINE",
    "POWER TRAIN",
    "STEERING",
    "SERVICE BRAKES",
    "FUEL/PROPULSION SYSTEM",
    "AIR BAGS",
    "DRIVER ASSISTANCE (ADAS)",
    "EXTERIOR LIGHTING",
    "VISIBILITY/WIPER",
    "STRUCTURE",
    "VEHICLE SPEED CONTROL",
    "SUSPENSION",
    "SEATS/SEAT BELTS",
]

_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{11,17}\b")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_EDITOR_TOKEN_RE = re.compile(r"\*[a-z]{1,3}\b|\btl\*|\btr\*|\bmr\b", re.IGNORECASE)
_REDACTION_RE = re.compile(r"\[?x{3,}\]?|\(xxx\)|\bxxxx+\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase a complaint narrative and strip out editorial tokens, redactions, VINs, phones, urls."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _REDACTION_RE.sub(" ", text)
    text = _EDITOR_TOKEN_RE.sub(" ", text)
    text = _VIN_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9/\-\s]", " ", text)  # keep alphanum, slash, hyphen
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def primary_label(components: str) -> str | None:
    """Map a raw comma-separated component string to a canonical class label."""
    if not isinstance(components, str) or not components.strip():
        return None
    first = components.split(",")[0].strip().upper()
    return COMPONENT_MAP.get(first)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw JSONL complaint dump into a DataFrame."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


MAX_PER_CLASS = 3500  # cap so the big classes don't dominate


def build_dataset(raw: pd.DataFrame, min_words: int = 8, max_per_class: int = MAX_PER_CLASS,
                  seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Clean text, derive labels, drop unusable rows, and cap class sizes."""
    df = raw.copy()
    df["label"] = df["components"].apply(primary_label)
    df = df[df["label"].isin(CANONICAL_LABELS)].copy()
    df["text"] = df["summary"].apply(clean_text)
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"] >= min_words].copy()
    df = df.drop_duplicates(subset=["text"])
    if max_per_class:
        parts = [
            grp if len(grp) <= max_per_class else grp.sample(max_per_class, random_state=seed)
            for _, grp in df.groupby("label", sort=False)
        ]
        df = pd.concat(parts, ignore_index=True)
    keep = [
        "text", "label", "make", "model", "modelYear",
        "crash", "fire", "numberOfInjuries", "numberOfDeaths", "n_words",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_splits(
    df: pd.DataFrame, seed: int = RANDOM_SEED, test_size: float = 0.15, val_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split on the component label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_frac = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROC_DIR / "train.csv", index=False)
    val.to_csv(PROC_DIR / "val.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved splits."""
    train = pd.read_csv(PROC_DIR / "train.csv")
    val = pd.read_csv(PROC_DIR / "val.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")
    return train, val, test


def save_committed_sample(df: pd.DataFrame, n: int = 400, seed: int = RANDOM_SEED) -> None:
    """Save a small class-balanced sample for the repo."""
    per = max(5, n // len(CANONICAL_LABELS))
    parts = [
        grp.sample(min(len(grp), per), random_state=seed)
        for _, grp in df.groupby("label", sort=False)
    ]
    sample = pd.concat(parts, ignore_index=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)


def get_class_names() -> list[str]:
    return list(CANONICAL_LABELS)


def build_and_save(raw_path: Path = RAW_PATH) -> dict:
    """Run the whole thing, from raw data to splits saved on disk."""
    raw = load_raw(raw_path)
    df = build_dataset(raw)
    train, val, test = make_splits(df)
    save_splits(train, val, test)
    save_committed_sample(df)
    return {
        "total": int(len(df)),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "n_classes": int(df["label"].nunique()),
        "class_counts": df["label"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    stats = build_and_save()
    print(json.dumps(stats, indent=2))
