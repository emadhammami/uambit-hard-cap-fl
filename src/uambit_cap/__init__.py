
# src/uambit_cap/__init__.py

"""
uambit_cap — package for uAMBIT + baselines (split from the single-cell notebook).

Convenient re-exports so you can write:
    from uambit_cap import RunConfig, run_all, run_e16, make_compressor
    from uambit_cap import UAmbitCompressor, TopKCompressor, STCCompressor, QSGDCompressor, EFSignCompressor
    from uambit_cap import build_lenet5, load_breastmnist
"""

# Experiments / runners
from .experiment import RunConfig, make_compressor, run_e16, run_all

# Compressors
from .compressors import (
    Compressor,
    UAmbitCompressor,
    TopKCompressor,
    STCCompressor,
    QSGDCompressor,
    EFSignCompressor,
)

# Model & data
from .model_data import build_lenet5, load_breastmnist

# Utilities
from .utils import (
    set_seeds,
    kb,
    flatten_weights,
    unflatten_weights,
    maybe_uint16,
    pack,
    unpack,
    ensure_dir,
)

__all__ = [
    # experiment
    "RunConfig", "make_compressor", "run_e16", "run_all",
    # compressors
    "Compressor", "UAmbitCompressor", "TopKCompressor", "STCCompressor",
    "QSGDCompressor", "EFSignCompressor",
    # model/data
    "build_lenet5", "load_breastmnist",
    # utils
    "set_seeds", "kb", "flatten_weights", "unflatten_weights",
    "maybe_uint16", "pack", "unpack", "ensure_dir",
]

__version__ = "0.1.0"
