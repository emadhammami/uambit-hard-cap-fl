# UAMBIT: Hard-Cap Federated Compression with Predictable Uplink

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emadhammami/uambit-hard-cap-fl/blob/main/uambit_budget_cap.ipynb)

**One-liner:** Enforce a strict per-round uplink budget (KiB/client/round) in FL using threshold search + sign+single-scale quantization + error-feedback (single gzip payload). Predictable bytes with competitive accuracy.

## TL;DR results (BreastMNIST, 0.5 KiB cap)
| Method | Uplink (KiB) | Std (bytes) | Test Acc |
|---|---:|---:|---:|
| **UAMBIT (ours)** | **0.499** | **0.8** | **0.7308** |
| Top-k (budgeted) | 0.498 | 1.0 | 0.7308 |
| STC (budgeted) | 0.499 | 0.6 | 0.7308 |
| QSGD-8 | 0.737 | 2.3 | 0.7308 |
| QSGD-4 | 0.720 | 2.9 | 0.7308 |
| EF-Sign | 0.590 | 1.5 | 0.7115 |

<p align="center">
  <img src="notebooks/plots/all_methods_ecdf_overlay.png" width="65%" alt="ECDF of realized bytes"/>
</p>

## Quickstart
Open the Colab and run end-to-end (dataset download, training, plots):
> **Colab:** `uambit_budget_cap.ipynb`

Local (optional):
```bash
pip install -r requirements.txt
python -m pip install jupyter  # if needed
jupyter notebook
