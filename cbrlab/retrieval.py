"""Embedding and similarity utilities with lightweight fallbacks."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

try:  # optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy not available in minimal envs
    np = None

try:  # optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - sentence-transformers optional
    SentenceTransformer = None  # type: ignore


class SimpleTfidf:
    """Tiny TF-IDF encoder that avoids heavy third-party deps."""

    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [tok.lower() for tok in text.split() if tok.strip()]

    def _build_vocab(self, docs: List[List[str]]) -> None:
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        n_docs = len(docs)
        self.vocab = sorted(df.keys())
        self.idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()
        }

    def _vectorize(self, tokens: List[str]) -> List[float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vec = [
            (counts.get(term, 0) / total) * self.idf.get(term, 0.0)
            for term in self.vocab
        ]
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def fit_transform(self, texts: Sequence[str]) -> List[List[float]]:
        docs = [self._tokenize(t) for t in texts]
        self._build_vocab(docs)
        return [self._vectorize(tokens) for tokens in docs]

    def transform(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vectorize(self._tokenize(t)) for t in texts]

    @staticmethod
    def similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return float(sum(a * b for a, b in zip(vec_a, vec_b)))


class Embedder:
    """Sentence-Transformer if available, otherwise a lightweight TF-IDF encoder."""

    def __init__(self) -> None:
        self.mode = "tfidf"
        self.vectorizer = SimpleTfidf()
        if SentenceTransformer is not None and np is not None:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.mode = "sbert"
            except Exception:  # pragma: no cover - optional model download failure
                self.model = None  # type: ignore

    def fit(self, texts: Sequence[str]) -> None:
        if self.mode == "sbert" and getattr(self, "model", None) is not None:
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        else:
            self.mode = "tfidf"
            self.embeddings = self.vectorizer.fit_transform(texts)

    def encode(self, texts: Sequence[str]) -> Sequence[Any]:
        if self.mode == "sbert" and getattr(self, "model", None) is not None:
            return self.model.encode(texts, normalize_embeddings=True)
        return self.vectorizer.transform(texts)

    def topk(self, query: str, k: int = 3) -> Tuple[List[int], List[float]]:
        q_vecs = self.encode([query])
        q_vec = q_vecs[0]
        if self.mode == "sbert" and np is not None:
            sims = (self.embeddings @ q_vec).tolist()
        else:
            sims = [self.vectorizer.similarity(q_vec, vec) for vec in self.embeddings]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return order, [float(sims[i]) for i in order]


def prepare_text(problem: Dict[str, Any], text_field: str | None) -> str:
    if text_field:
        return text_field
    parts = [f"{k}={v}" for k, v in problem.items()]
    return "; ".join(parts)

