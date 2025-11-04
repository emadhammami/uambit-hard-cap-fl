# UAMBIT: Hard-Cap Federated Compression with Predictable Uplink

UAMBIT enforces a strict per-round uplink budget (KiB/client/round) via threshold search + sign+single-scale quantization + error feedback (one gzip payload). Predictable, SLA-friendly bytes with competitive accuracy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emadhammami/uambit-hard-cap-fl/blob/main/notebooks/uambit_budget_cap.ipynb)

## Quickstart
```bash
pip install -r requirements.txt
# Or open the Colab notebook above
