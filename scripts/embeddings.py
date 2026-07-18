"""Pretrained GloVe word embeddings for the TextCNN (transfer learning).

Initialising the embedding layer with GloVe vectors (Pennington et al., 2014)
injects general-language knowledge the model can't learn from ~26k complaints
alone, which lifts accuracy and — importantly — the low-data (cold-start) regime.

The vectors are downloaded once to ``data/raw/`` and are only needed at *training*
time; the learned matrix is baked into the saved ``.pt`` so the deployed app stays
self-contained. GloVe is distributed under the Public Domain Dedication and License.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
GLOVE_DIM = 200
GLOVE_PATH = BASE_DIR / "data" / "raw" / f"glove.6B.{GLOVE_DIM}d.txt"
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


def ensure_glove(path: Path = GLOVE_PATH) -> Path | None:
    """Download + extract GloVe 6B if missing. Returns the path, or None on failure."""
    if path.exists():
        return path
    try:
        import urllib.request

        zip_path = path.parent / "glove.6B.zip"
        print(f"Downloading GloVe (~822 MB) -> {zip_path} ...", flush=True)
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(path.name, path.parent)
        zip_path.unlink(missing_ok=True)
        return path if path.exists() else None
    except Exception as err:  # noqa: BLE001
        print(f"GloVe download failed ({err}); falling back to scratch embeddings.")
        return None


_GLOVE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def load_glove(path: Path = GLOVE_PATH, dim: int = GLOVE_DIM) -> dict[str, np.ndarray]:
    """Load a GloVe text file into a {word: vector} dict (cached in-process).

    The full pipeline builds the embedding matrix several times (once per model
    and once per learning-curve point); caching avoids re-reading the ~700 MB file.
    """
    key = str(path)
    if key in _GLOVE_CACHE:
        return _GLOVE_CACHE[key]
    vectors: dict[str, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    _GLOVE_CACHE[key] = vectors
    return vectors


def build_embedding_matrix(vocab: dict[str, int], dim: int = GLOVE_DIM,
                           path: Path = GLOVE_PATH, seed: int = 42) -> tuple[np.ndarray, float]:
    """Build a (vocab_size, dim) matrix; GloVe where available, small-random for OOV.

    Returns the matrix and the fraction of the vocabulary covered by GloVe.
    """
    glove = load_glove(path, dim)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0  # <pad>
    hits = 0
    for tok, idx in vocab.items():
        vec = glove.get(tok)
        if vec is not None:
            matrix[idx] = vec
            hits += 1
    coverage = hits / max(1, len(vocab) - 2)  # exclude <pad>, <unk>
    return matrix, coverage
