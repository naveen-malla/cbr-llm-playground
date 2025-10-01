from typing import List, Dict, Any, Tuple
import numpy as np

# Optional embedding backend
class Embedder:
    def __init__(self):
        self.mode = "tfidf"
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "sbert"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer()
            self.mode = "tfidf"

    def fit(self, texts: List[str]):
        if self.mode == "sbert":
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        else:
            self.embeddings = self.vectorizer.fit_transform(texts)

    def encode(self, texts: List[str]):
        if self.mode == "sbert":
            return self.model.encode(texts, normalize_embeddings=True)
        else:
            return self.vectorizer.transform(texts)

    def topk(self, query: str, k: int = 3) -> List[int]:
        q = self.encode([query])
        if self.mode == "sbert":
            sims = (self.embeddings @ q[0]).tolist()
        else:
            sims = (self.embeddings @ q.T).toarray().ravel().tolist()
        idx = np.argsort(sims)[::-1][:k]
        return idx.tolist(), [sims[i] for i in idx]

def prepare_text(problem: Dict[str, Any], text_field: str | None) -> str:
    if text_field:
        return text_field
    # Build a compact text from A/V features
    parts = [f"{k}={v}" for k, v in problem.items()]
    return "; ".join(parts)
