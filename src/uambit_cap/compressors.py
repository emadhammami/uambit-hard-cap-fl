
# src/uambit_cap/compressors.py

import numpy as np
from typing import List
from .utils import flatten_weights, unflatten_weights, maybe_uint16, pack, unpack

class Compressor:
    def __init__(self):
        self.residual = None  # error-feedback buffer

    def _apply_ef(self, grad: np.ndarray) -> np.ndarray:
        if self.residual is None:
            self.residual = np.zeros_like(grad, dtype=np.float32)
        return grad + self.residual

    def _update_residual(self, grad: np.ndarray, rec: np.ndarray):
        if self.residual is None:
            self.residual = np.zeros_like(grad, dtype=np.float32)
        self.residual = grad - rec

    def compress(self, delta_weights: List[np.ndarray], target_kib: float, **kwargs) -> bytes:
        raise NotImplementedError

    def decompress(self, blob: bytes) -> List[np.ndarray]:
        raise NotImplementedError


# ----------------------------- uAMBIT (budgeted: threshold + sign+scale + EF + gzip) -----------------------------
class UAmbitCompressor(Compressor):
    def __init__(self, enable_ef=True):
        super().__init__()
        self.enable_ef = enable_ef
        self._shapes = None

    def compress(self, delta_weights: List[np.ndarray], target_kib: float, **kwargs) -> bytes:
        flat, shapes = flatten_weights(delta_weights)
        self._shapes = shapes
        grad = flat.astype(np.float32)
        if self.enable_ef:
            grad = self._apply_ef(grad)
        lo, hi, best = 0.0, float(np.max(np.abs(grad)) + 1e-12), None
        for _ in range(24):
            tau = 0.5 * (lo + hi)
            mask = np.abs(grad) >= tau
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                break
            val = grad[mask]
            scale = np.max(np.abs(val)).astype(np.float32) if val.size else np.float32(1.0)
            signs = (val >= 0).astype(np.int8)
            payload = {
                "type": "uambit",
                "idx": maybe_uint16(idx),
                "signs": signs,
                "scale": np.float32(scale),
                "shapes": shapes,
                "N": int(grad.size),
            }
            blob = pack(payload)
            if (len(blob) / 1024.0) <= target_kib:
                best, hi = payload, tau
            else:
                lo = tau
        if best is None:
            best = {
                "type": "uambit",
                "idx": np.array([], dtype=np.uint16),
                "signs": np.array([], dtype=np.int8),
                "scale": np.float32(0.0),
                "shapes": shapes,
                "N": int(grad.size),
            }
        return pack(best)

    def decompress(self, blob: bytes) -> List[np.ndarray]:
        p = unpack(blob)
        N = p["N"]
        grad_hat = np.zeros(N, dtype=np.float32)
        if p["idx"].size:
            val = (p["signs"].astype(np.float32) * 2.0 - 1.0) * p["scale"]
            grad_hat[p["idx"].astype(np.int64)] = val
        return unflatten_weights(grad_hat, p["shapes"])


# ----------------------------- Top-k / DGC (budgeted, full-precision values) -----------------------------
class TopKCompressor(Compressor):
    def __init__(self, enable_ef=True):
        super().__init__()
        self.enable_ef = enable_ef

    def compress(self, delta_weights: List[np.ndarray], target_kib: float, **kwargs) -> bytes:
        flat, shapes = flatten_weights(delta_weights)
        grad = flat.astype(np.float32)
        if self.enable_ef:
            grad = self._apply_ef(grad)
        lo, hi, best = 0.0, float(np.max(np.abs(grad)) + 1e-12), None
        for _ in range(24):
            tau = 0.5 * (lo + hi)
            mask = np.abs(grad) >= tau
            idx = np.nonzero(mask)[0]
            val = grad[mask].astype(np.float32)
            payload = {
                "type": "topk",
                "idx": maybe_uint16(idx),
                "val": val,
                "shapes": shapes,
                "N": int(grad.size),
            }
            blob = pack(payload)
            if (len(blob) / 1024.0) <= target_kib:
                best, hi = payload, tau
            else:
                lo = tau
        if best is None:
            best = {
                "type": "topk",
                "idx": np.array([], dtype=np.uint16),
                "val": np.array([], dtype=np.float32),
                "shapes": shapes,
                "N": int(grad.size),
            }
        return pack(best)

    def decompress(self, blob: bytes) -> List[np.ndarray]:
        p = unpack(blob)
        N = p["N"]
        grad_hat = np.zeros(N, dtype=np.float32)
        if p["idx"].size:
            grad_hat[p["idx"].astype(np.int64)] = p["val"].astype(np.float32)
        return unflatten_weights(grad_hat, p["shapes"])


# ----------------------------- STC (budgeted sparse ternary) -----------------------------
class STCCompressor(Compressor):
    def __init__(self, enable_ef=True):
        super().__init__()
        self.enable_ef = enable_ef

    def compress(self, delta_weights: List[np.ndarray], target_kib: float, **kwargs) -> bytes:
        flat, shapes = flatten_weights(delta_weights)
        grad = flat.astype(np.float32)
        if self.enable_ef:
            grad = self._apply_ef(grad)
        lo, hi, best = 0.0, float(np.max(np.abs(grad)) + 1e-12), None
        for _ in range(24):
            tau = 0.5 * (lo + hi)
            mask = np.abs(grad) >= tau
            idx = np.nonzero(mask)[0]
            sel = grad[mask].astype(np.float32)
            scale = np.mean(np.abs(sel)).astype(np.float32) if sel.size else np.float32(0.0)
            signs = (sel >= 0).astype(np.int8)
            payload = {
                "type": "stc",
                "idx": maybe_uint16(idx),
                "signs": signs,
                "scale": np.float32(scale),
                "shapes": shapes,
                "N": int(grad.size),
            }
            blob = pack(payload)
            if (len(blob) / 1024.0) <= target_kib:
                best, hi = payload, tau
            else:
                lo = tau
        if best is None:
            best = {
                "type": "stc",
                "idx": np.array([], dtype=np.uint16),
                "signs": np.array([], dtype=np.int8),
                "scale": np.float32(0.0),
                "shapes": shapes,
                "N": int(grad.size),
            }
        return pack(best)

    def decompress(self, blob: bytes) -> List[np.ndarray]:
        p = unpack(blob)
        N = p["N"]
        grad_hat = np.zeros(N, dtype=np.float32)
        if p["idx"].size:
            val = (p["signs"].astype(np.float32) * 2.0 - 1.0) * p["scale"]
            grad_hat[p["idx"].astype(np.int64)] = val
        return unflatten_weights(grad_hat, p["shapes"])


# ----------------------------- QSGD (dense; 8/4-bit) -----------------------------
class QSGDCompressor(Compressor):
    def __init__(self, bits: int = 8, enable_ef=True, mask_ratio: float = 1.0):
        super().__init__()
        assert bits in (8, 4)
        self.bits = bits
        self.enable_ef = enable_ef
        self.mask_ratio = mask_ratio

    def compress(self, delta_weights: List[np.ndarray], target_kib: float, **kwargs) -> bytes:
        flat, shapes = flatten_weights(delta_weights)
        grad = flat.astype(np.float32)
        if self.enable_ef:
            grad = self._apply_ef(grad)
        N = grad.size
        if self.mask_ratio < 1.0:
            m = max(1, int(self.mask_ratio * N))
            idx = np.random.choice(N, size=m, replace=False)
            sel = grad[idx]
        else:
            idx = np.arange(N, dtype=np.int64)
            sel = grad
        s = (1 << self.bits) - 1
        scale = np.max(np.abs(sel)).astype(np.float32) + 1e-12
        q = np.clip(np.round((np.abs(sel) / scale) * s), 0, s).astype(np.uint8)
        signs = (sel >= 0).astype(np.int8)
        payload = {
            "type": "qsgd",
            "bits": self.bits,
            "idx": maybe_uint16(idx),
            "q": q,
            "signs": signs,
            "scale": np.float32(scale),
            "shapes": shapes,
            "N": int(N),
        }
        return pack(payload)

    def decompress(self, blob: bytes) -> List[np.ndarray]:
        p = unpack(blob)
        N = p["N"]
        grad_hat = np.zeros(N, dtype=np.float32)
        s = (1 << int(p["bits"])) - 1
        if p["idx"].size:
            mag = (p["q"].astype(np.float32) / s) * p["scale"].astype(np.float32)
            val = (p["signs"].astype(np.float32) * 2.0 - 1.0) * mag
            grad_hat[p["idx"].astype(np.int64)] = val
        return unflatten_weights(grad_hat, p["shapes"])


# ----------------------------- EF-Sign (dense; 1-bit with scale) -----------------------------
class EFSignCompressor(Compressor):
    def __init__(self, enable_ef=True, mask_ratio: float = 1.0):
        super().__init__()
        self.enable_ef = enable_ef
        self.mask_ratio = mask_ratio

    def compress(self, delta_weights: List[np.ndarray], target_kib: float, **kwargs) -> bytes:
        flat, shapes = flatten_weights(delta_weights)
        grad = flat.astype(np.float32)
        if self.enable_ef:
            grad = self._apply_ef(grad)
        N = grad.size
        if self.mask_ratio < 1.0:
            m = max(1, int(self.mask_ratio * N))
            idx = np.random.choice(N, size=m, replace=False)
            sel = grad[idx]
        else:
            idx = np.arange(N, dtype=np.int64)
            sel = grad
        scale = np.mean(np.abs(sel)).astype(np.float32) if sel.size else np.float32(0.0)
        signs = (sel >= 0).astype(np.int8)
        payload = {
            "type": "efsign",
            "idx": maybe_uint16(idx),
            "signs": signs,
            "scale": np.float32(scale),
            "shapes": shapes,
            "N": int(N),
        }
        return pack(payload)

    def decompress(self, blob: bytes) -> List[np.ndarray]:
        p = unpack(blob)
        N = p["N"]
        grad_hat = np.zeros(N, dtype=np.float32)
        if p["idx"].size:
            val = (p["signs"].astype(np.float32) * 2.0 - 1.0) * p["scale"].astype(np.float32)
            grad_hat[p["idx"].astype(np.int64)] = val
        return unflatten_weights(grad_hat, p["shapes"])
