
# src/uambit_cap/utils.py

import os
import io
import gzip
import pickle
import random
from typing import List, Tuple

import numpy as np


def set_seeds(seed: int = 0):
    """Set Python, NumPy, and TensorFlow RNG seeds (TF optional)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # lazy import so this file has no hard TF dep at import time
        tf.random.set_seed(seed)
    except Exception:
        # If TF isn't available (e.g., during lightweight tooling), just skip.
        pass


def kb(blob: bytes) -> float:
    """Return size in KiB for a given bytes object."""
    return len(blob) / 1024.0


def flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple]]:
    """Flatten a list of ndarrays into one 1D float32 vector and record shapes."""
    flat, shapes = [], []
    for w in weights:
        shapes.append(w.shape)
        flat.append(w.reshape(-1))
    return np.concatenate(flat).astype(np.float32), shapes


def unflatten_weights(flat_vec: np.ndarray, shapes: List[Tuple]) -> List[np.ndarray]:
    """Reconstruct list of ndarrays from a flat vector and shapes."""
    out, idx = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat_vec[idx:idx + n].reshape(s))
        idx += n
    return out


def maybe_uint16(idx: np.ndarray) -> np.ndarray:
    """Store indices as uint16 when possible, else uint32 (to mirror the notebook behavior)."""
    return idx.astype(np.uint16) if idx.size < (1 << 16) else idx.astype(np.uint32)


def pack(payload: dict) -> bytes:
    """gzip + pickle payload with the same settings as the notebook."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=3) as f:
        f.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return buf.getvalue()


def unpack(blob: bytes) -> dict:
    """Inverse of pack()."""
    with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as f:
        return pickle.loads(f.read())


def ensure_dir(p: str):
    """Create directory if it does not exist."""
    os.makedirs(p, exist_ok=True)
