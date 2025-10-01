import argparse
from pathlib import Path
import numpy as np
try:
    import faiss
except Exception:
    faiss = None

class SimpleIndex:
    def __init__(self):
        self.X = None

    def build(self, X):
        self.X = np.asarray(X, dtype="float32")

    def search(self, q, k=5):
        X = self.X
        sims = X @ q
        idx = np.argsort(sims)[::-1][:k]
        return idx, sims[idx]

def build_index(embs, out_path: str | Path):
    idx = SimpleIndex()
    idx.build(embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9))
    Path(out_path).write_bytes(b"placeholder")  # minimal placeholder; in-memory for demo
    return idx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    args = ap.parse_args()
    print("Index build stub complete.")
