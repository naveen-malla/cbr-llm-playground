import numpy as np

def fuse(symbolic_scores, vector_scores, alpha: float = 0.5):
    # assume both are lists aligned by candidate order
    s = np.array(symbolic_scores)
    v = np.array(vector_scores)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    fused = alpha * s + (1 - alpha) * v
    order = np.argsort(fused)[::-1]
    return order.tolist(), fused[order].tolist()
