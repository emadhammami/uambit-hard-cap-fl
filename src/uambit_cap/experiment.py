
# src/uambit_cap/experiment.py

import os, glob
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .utils import set_seeds, flatten_weights, ensure_dir
from .compressors import (
    Compressor,
    UAmbitCompressor, TopKCompressor, STCCompressor,
    QSGDCompressor, EFSignCompressor
)
from .model_data import build_lenet5, load_breastmnist


@dataclass
class RunConfig:
    rounds: int = 15
    local_epochs: int = 1
    batch_size: int = 128
    clients_per_round: int = 3
    budget_kib: float = 0.5
    method: str = "uambit"  # uambit|topk|stc|qsgd8|qsgd4|efsign
    mask_ratio: float = 1.0
    outdir: str = "plots"
    no_plots: bool = False


def make_compressor(cfg: RunConfig) -> Compressor:
    if cfg.method == "uambit":
        return UAmbitCompressor(True)
    if cfg.method == "topk":
        return TopKCompressor(True)
    if cfg.method == "stc":
        return STCCompressor(True)
    if cfg.method == "qsgd8":
        return QSGDCompressor(bits=8, enable_ef=True, mask_ratio=cfg.mask_ratio)
    if cfg.method == "qsgd4":
        return QSGDCompressor(bits=4, enable_ef=True, mask_ratio=cfg.mask_ratio)
    if cfg.method == "efsign":
        return EFSignCompressor(True, mask_ratio=cfg.mask_ratio)
    raise ValueError(cfg.method)


# ----------------------------- Plot helpers -----------------------------
def plot_curves(val_hist, mean_bytes_kib, cfg: RunConfig):
    ensure_dir(cfg.outdir)
    acc_path = os.path.join(cfg.outdir, f"{cfg.method}_E16_b{cfg.budget_kib:.2f}KiB_acc.png")
    plt.figure()
    plt.plot(range(1, len(val_hist) + 1), val_hist)
    plt.xlabel("Round")
    plt.ylabel("Val Acc")
    plt.title(f"BreastMNIST — {cfg.method} — Acc vs Rounds")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    bytes_path = os.path.join(cfg.outdir, f"{cfg.method}_E16_b{cfg.budget_kib:.2f}KiB_bytes.png")
    plt.figure()
    plt.plot(range(1, len(mean_bytes_kib) + 1), mean_bytes_kib)
    plt.xlabel("Round")
    plt.ylabel("Mean Uplink (KiB)")
    plt.title(f"BreastMNIST — {cfg.method} — Mean Uplink vs Rounds")
    plt.tight_layout()
    plt.savefig(bytes_path, dpi=150)
    plt.close()

    return [acc_path, bytes_path]


def plot_ecdf(all_round_bytes, cfg: RunConfig):
    ensure_dir(cfg.outdir)
    flat = np.array([b for round_list in all_round_bytes for b in round_list], dtype=np.float64) / 1024.0
    if flat.size == 0:
        return None
    flat.sort()
    y = np.arange(1, flat.size + 1) / flat.size
    ecdf_path = os.path.join(cfg.outdir, f"{cfg.method}_E16_b{cfg.budget_kib:.2f}KiB_ecdf.png")
    plt.figure()
    plt.plot(flat, y)
    plt.xlabel("Per-client Uplink (KiB)")
    plt.ylabel("ECDF")
    plt.title(f"BreastMNIST — {cfg.method} — Uplink ECDF")
    plt.tight_layout()
    plt.savefig(ecdf_path, dpi=150)
    plt.close()
    return ecdf_path


# ----------------------------- Main experiment (E16) -----------------------------
def run_e16(cfg: RunConfig):
    set_seeds(0)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_breastmnist()

    K = cfg.clients_per_round
    chunk = len(x_train) // K
    clients = [(x_train[i * chunk:(i + 1) * chunk], y_train[i * chunk:(i + 1) * chunk]) for i in range(K)]

    compressors = [make_compressor(cfg) for _ in range(K)]
    global_model = build_lenet5(input_shape=x_train.shape[1:], num_classes=n_classes)
    global_weights = global_model.get_weights()

    mean_uplink_bytes_per_round, val_hist, all_round_bytes = [], [], []

    for r in range(cfg.rounds):
        deltas, round_bytes = [], []
        for k in range(K):
            local = build_lenet5(input_shape=x_train.shape[1:], num_classes=n_classes)
            local.set_weights(global_weights)
            local.fit(
                clients[k][0], clients[k][1],
                epochs=cfg.local_epochs, batch_size=cfg.batch_size,
                verbose=0, validation_data=(x_val, y_val),
                callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
            )
            lw = local.get_weights()
            delta = [p - g for p, g in zip(lw, global_weights)]
            comp = compressors[k]
            blob = comp.compress(delta, target_kib=cfg.budget_kib)
            round_bytes.append(len(blob))
            rec = comp.decompress(blob)
            flat_grad, _ = flatten_weights(delta)
            flat_rec, _ = flatten_weights(rec)
            comp._update_residual(flat_grad, flat_rec)
            deltas.append(rec)
            del local
            tf.keras.backend.clear_session()

        agg = [np.mean([d[i] for d in deltas], axis=0) for i in range(len(global_weights))]
        global_weights = [g + a for g, a in zip(global_weights, agg)]
        global_model.set_weights(global_weights)

        val_loss, val_acc = global_model.evaluate(x_val, y_val, verbose=0)
        mean_uplink_bytes_per_round.append(np.mean(round_bytes))
        all_round_bytes.append(round_bytes.copy())
        val_hist.append(float(val_acc))
        print(f"[Round {r+1:02d}] Val acc={val_acc:.4f} | mean uplink={np.mean(round_bytes)/1024:.3f} KiB (std {np.std(round_bytes)/1024:.3f})")

    test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest acc={test_acc:.4f}")
    print(
        "Per-round uplink bytes: mean={:.1f}, std={:.1f} (KiB mean={:.3f})".format(
            float(np.mean(mean_uplink_bytes_per_round)),
            float(np.std(mean_uplink_bytes_per_round)),
            float(np.mean(mean_uplink_bytes_per_round) / 1024.0),
        )
    )

    # Plots
    paths = plot_curves(val_hist, [b / 1024.0 for b in mean_uplink_bytes_per_round], cfg)
    ecdf = plot_ecdf(all_round_bytes, cfg)
    if ecdf:
        paths.append(ecdf)
    print("Saved plots:", *paths, sep="\n  ")
    try:
        from IPython.display import display, Image as IPyImage
        for p in paths[:3]:  # preview up to 3
            display(IPyImage(filename=p))
    except Exception:
        pass

    return {
        "val_acc_hist": val_hist,
        "test_acc": float(test_acc),
        "uplink_bytes_mean": float(np.mean(mean_uplink_bytes_per_round)),
        "uplink_bytes_std": float(np.std(mean_uplink_bytes_per_round)),
        "plot_paths": paths,
    }


def run_all(methods=("uambit", "topk", "stc", "qsgd8", "qsgd4", "efsign"),
            budget_kib=0.5, rounds=15, clients=3, mask_ratio_dense=None):
    results = {}
    for m in methods:
        print("\n==============================")
        print(f"Running {m} @ {budget_kib:.2f} KiB")
        print("==============================")
        cfg = RunConfig(
            rounds=rounds, local_epochs=1, batch_size=128,
            clients_per_round=clients, budget_kib=budget_kib,
            method=m,
            mask_ratio=(mask_ratio_dense if (mask_ratio_dense and m in {"qsgd8", "qsgd4", "efsign"}) else 1.0),
            outdir="plots", no_plots=False
        )
        results[m] = run_e16(cfg)
    return results
