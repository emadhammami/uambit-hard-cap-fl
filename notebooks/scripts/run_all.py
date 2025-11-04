
# notebooks/scripts/run_all.py

# If running in a fresh Colab-like environment, first install deps in a cell:
# %pip -q install -U "tensorflow>=2.15,<2.19" medmnist matplotlib numpy

import glob
from uambit_cap import run_all

if __name__ == "__main__":
    _ = run_all(
        methods=("uambit", "topk", "stc", "qsgd8", "qsgd4", "efsign"),
        budget_kib=0.5,
        rounds=50,      # or 15 if you want quick
        clients=3,
        mask_ratio_dense=0.002  # ~0.1–0.5% of params; tweak until ~0.5 KiB
    )

    print("\nAll plots saved in ./plots :")
    for p in sorted(glob.glob("plots/*.png")):
        print(" ", p)
