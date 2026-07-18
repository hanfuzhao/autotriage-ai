"""Three models behind one interface: a naive baseline, TF-IDF + LogReg, and a TextCNN (Kim 2014)."""
from __future__ import annotations

import os

# avoids the duplicate OpenMP crash with torch plus sklearn on mac
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


class NaiveBaseline:
    """Always predicts the most frequent class, our accuracy floor."""

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.priors_: np.ndarray | None = None
        self.majority_: str | None = None

    def fit(self, texts, labels) -> "NaiveBaseline":
        counts = Counter(labels)
        self.classes_ = sorted(counts)
        total = sum(counts.values())
        self.priors_ = np.array([counts[c] / total for c in self.classes_])
        self.majority_ = max(counts, key=counts.get)
        return self

    def predict(self, texts) -> np.ndarray:
        return np.array([self.majority_] * len(list(texts)))

    def predict_proba(self, texts) -> np.ndarray:
        n = len(list(texts))
        return np.tile(self.priors_, (n, 1))

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path) -> "NaiveBaseline":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class ClassicalModel:
    """TF-IDF n-grams into a class-weighted LogReg, fast and interpretable."""

    def __init__(self, C: float = 3.0, max_features: int = 40000, ngram_max: int = 2) -> None:
        self.C = C
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, ngram_max),
                        min_df=3,
                        max_df=0.9,
                        sublinear_tf=True,
                        max_features=max_features,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    @property
    def classes_(self) -> list[str]:
        return list(self.pipeline.named_steps["clf"].classes_)

    def fit(self, texts, labels) -> "ClassicalModel":
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts) -> np.ndarray:
        return self.pipeline.predict(list(texts))

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))

    def top_features(self, label: str, k: int = 12) -> list[tuple[str, float]]:
        """Return the k tokens most indicative of a given class."""
        clf = self.pipeline.named_steps["clf"]
        vocab = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        idx = list(clf.classes_).index(label)
        coefs = clf.coef_[idx]
        top = np.argsort(coefs)[::-1][:k]
        return [(vocab[i], float(coefs[i])) for i in top]

    def explain(self, text: str, k: int = 8) -> list[tuple[str, float]]:
        """Token contributions toward the predicted class for one input."""
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        vec = tfidf.transform([text])
        pred_idx = int(np.argmax(clf.decision_function(vec)))
        coefs = clf.coef_[pred_idx]
        vocab = tfidf.get_feature_names_out()
        contribs = vec.multiply(coefs).tocoo()
        pairs = [(vocab[j], float(v)) for j, v in zip(contribs.col, contribs.data)]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:k]

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path) -> "ClassicalModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")
PAD, UNK = "<pad>", "<unk>"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _select_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TextCNNModel:
    """TextCNN classifier over word embeddings, TextCNN, Kim 2014."""

    def __init__(
        self,
        embed_dim: int = 200,
        num_filters: int = 160,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.4,
        max_len: int = 220,
        min_freq: int = 2,
        max_vocab: int = 25000,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 30,
        patience: int = 5,
        pretrained: bool = True,
        seed: int = 42,
    ) -> None:
        self.cfg = dict(
            embed_dim=embed_dim, num_filters=num_filters, kernel_sizes=list(kernel_sizes),
            dropout=dropout, max_len=max_len, min_freq=min_freq, max_vocab=max_vocab,
            lr=lr, batch_size=batch_size, max_epochs=max_epochs, patience=patience,
            pretrained=pretrained, seed=seed,
        )
        self._emb_matrix = None
        self.glove_coverage: float | None = None
        self.vocab: dict[str, int] = {}
        self.classes_: list[str] = []
        self.net = None
        self.device = None
        self.history: list[dict] = []

    def _build_vocab(self, texts) -> None:
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        vocab = {PAD: 0, UNK: 1}
        for tok, freq in counter.most_common(self.cfg["max_vocab"]):
            if freq >= self.cfg["min_freq"]:
                vocab[tok] = len(vocab)
        self.vocab = vocab

    def _encode(self, text: str) -> list[int]:
        max_len = self.cfg["max_len"]
        ids = [self.vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def _batch_tensor(self, texts):
        import torch

        arr = np.array([self._encode(t) for t in texts], dtype=np.int64)
        return torch.from_numpy(arr)

    def _maybe_load_pretrained(self) -> None:
        """Build a GloVe-initialised embedding matrix for the vocab."""
        if not self.cfg.get("pretrained"):
            return
        try:
            from scripts.embeddings import GLOVE_DIM, build_embedding_matrix, ensure_glove
        except Exception:  # noqa: BLE001
            return
        if self.cfg["embed_dim"] != GLOVE_DIM:
            return
        if ensure_glove() is None:
            return
        matrix, coverage = build_embedding_matrix(self.vocab, dim=GLOVE_DIM, seed=self.cfg["seed"])
        self._emb_matrix = matrix
        self.glove_coverage = coverage

    def _build_net(self, n_classes: int):
        import torch
        import torch.nn as nn

        cfg = self.cfg
        emb_matrix = self._emb_matrix

        class _Net(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, cfg["embed_dim"], padding_idx=0)
                if emb_matrix is not None:
                    self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
                    self.embed.weight.data[0].zero_()
                self.convs = nn.ModuleList(
                    [nn.Conv1d(cfg["embed_dim"], cfg["num_filters"], k) for k in cfg["kernel_sizes"]]
                )
                self.dropout = nn.Dropout(cfg["dropout"])
                self.fc = nn.Linear(cfg["num_filters"] * len(cfg["kernel_sizes"]), n_classes)

            def forward(self, x):
                import torch
                import torch.nn.functional as F

                emb = self.embed(x).transpose(1, 2)
                feats = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
                out = self.dropout(torch.cat(feats, dim=1))
                return self.fc(out)

        return _Net(len(self.vocab))

    def fit(self, texts, labels, val_texts=None, val_labels=None, verbose: bool = True) -> "TextCNNModel":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        self.device = _select_device()
        texts, labels = list(texts), list(labels)

        self._build_vocab(texts)
        self._maybe_load_pretrained()
        self.classes_ = sorted(set(labels))
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([cls_to_idx[c] for c in labels])

        # class weights for imbalance
        counts = np.bincount(y, minlength=len(self.classes_)).astype(np.float64)
        weights = (counts.sum() / (len(counts) * np.clip(counts, 1, None))).astype(np.float32)

        self.net = self._build_net(len(self.classes_)).to(self.device)
        X = self._batch_tensor(texts)
        yt = torch.from_numpy(y)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=self.device))

        n = len(texts)
        bs = self.cfg["batch_size"]
        best_f1, best_state, patience_left = -1.0, None, self.cfg["patience"]

        for epoch in range(self.cfg["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = X[idx].to(self.device)
                yb = yt[idx].to(self.device)
                opt.zero_grad()
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach()) * len(idx)

            if val_texts is not None:
                val_pred = self.predict(val_texts)
                f1 = f1_score(val_labels, val_pred, average="macro")
            else:
                f1 = -total_loss / n
            self.history.append({"epoch": epoch, "train_loss": total_loss / n, "val_macro_f1": float(f1)})
            if verbose:
                print(f"    epoch {epoch:2d}  loss={total_loss / n:.4f}  val_macroF1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                patience_left = self.cfg["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print(f"    early stop at epoch {epoch} (best val macro-F1={best_f1:.4f})")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _logits(self, texts):
        import torch

        self.net.eval()
        outs = []
        bs = 256
        texts = list(texts)
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                xb = self._batch_tensor(texts[start:start + bs]).to(self.device)
                outs.append(self.net(xb).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(self.classes_)))

    def predict_proba(self, texts) -> np.ndarray:
        logits = self._logits(texts)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts) -> np.ndarray:
        proba = self.predict_proba(texts)
        idx = proba.argmax(axis=1)
        return np.array([self.classes_[i] for i in idx])

    def explain(self, text: str, k: int = 8, max_tokens: int = 80) -> list[tuple[str, float]]:
        """Leave-one-token-out saliency for the predicted class."""
        toks = tokenize(text)[:max_tokens]
        if not toks:
            return []
        base = self.predict_proba([text])[0]
        pred = int(base.argmax())
        perturbed = [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
        probs = self.predict_proba(perturbed)[:, pred]
        scores: dict[str, float] = {}
        for tok, p in zip(toks, probs):
            scores[tok] = max(scores.get(tok, 0.0), float(base[pred] - p))
        return sorted(scores.items(), key=lambda p: p[1], reverse=True)[:k]

    def save(self, path) -> None:
        import torch

        torch.save(
            {
                "cfg": self.cfg,
                "vocab": self.vocab,
                "classes": self.classes_,
                "state_dict": self.net.state_dict(),
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "TextCNNModel":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**{k: (tuple(v) if k == "kernel_sizes" else v) for k, v in blob["cfg"].items()})
        model.vocab = blob["vocab"]
        model.classes_ = blob["classes"]
        model.history = blob.get("history", [])
        model.device = torch.device("cpu")
        model.net = model._build_net(len(model.classes_)).to(model.device)
        model.net.load_state_dict(blob["state_dict"])
        model.net.eval()
        return model


MODEL_REGISTRY = {
    "naive": NaiveBaseline,
    "classical": ClassicalModel,
    "deep": TextCNNModel,
}
