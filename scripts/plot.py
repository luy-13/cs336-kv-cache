"""从 metrics 重新生成两张图，保证图可复现。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")


def _latest(pattern: str) -> Path | None:
    files = sorted(METRICS_DIR.glob(pattern))
    return files[-1] if files else None


def plot_training(metrics_path: Path, out_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    losses = metrics["loss"]["train"]
    val_steps = metrics["loss"]["val_steps"]
    val_losses = metrics["loss"]["val"]

    window = 20
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(7.2, 4.2), dpi=160)
    plt.plot(range(1, len(losses) + 1), losses, color="#1f77b4", linewidth=0.6, alpha=0.3)
    plt.plot(range(window, len(losses) + 1), smooth, color="#1f77b4", linewidth=1.8, label="Train (EMA-20)")
    plt.plot(val_steps, val_losses, "o-", color="#d62728", linewidth=1.5, markersize=3.5, label="Validation")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("TinyStories Pre-training: Train vs. Validation Loss")
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bench(csv_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    Ls = [int(r["L"]) for r in rows]
    num = lambda key: [float(r[key]) for r in rows]  # noqa: E731

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)

    ax1.plot(Ls, num("no_cache_ms"), "o--", color="#d62728", label="No Cache (Recompute)")
    ax1.plot(Ls, num("cache_ms"), "s-", color="#1f77b4", label="With KV Cache (Incremental)")
    ax1.set_xlabel("Generated Sequence Length (Tokens)")
    ax1.set_ylabel("End-to-End Latency (ms)")
    ax1.set_title("Decoding Latency")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(Ls, num("theory_kv_mb"), "k--", label="Theoretical Formula")
    ax2.plot(Ls, num("tensor_kv_mb"), "o-", color="#2ca02c", label="KV Tensor Memory")
    ax2.plot(Ls, num("peak_kv_delta_mb"), "^-", color="#d62728", label="Allocator Peak Delta")
    ax2.set_xlabel("Generated Sequence Length (Tokens)")
    ax2.set_ylabel("KV Cache Footprint (MB)")
    ax2.set_title("KV Cache Memory: Theory vs. Tensor vs. Allocator Peak")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from metrics")
    parser.add_argument("--train-metrics", type=Path, default=None)
    parser.add_argument("--bench-csv", type=Path, default=None)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_metrics = args.train_metrics or _latest("train_*.json")
    bench_csv = args.bench_csv or _latest("bench_*.csv")
    if train_metrics is None or bench_csv is None:
        raise SystemExit("找不到 metrics，请先运行 run_train.py 与 run_bench.py")

    plot_training(train_metrics, FIGURES_DIR / "training_loss.png")
    plot_bench(bench_csv, FIGURES_DIR / "kv_cache_benchmark.png")
    print(f"[Done] {FIGURES_DIR / 'training_loss.png'}")
    print(f"[Done] {FIGURES_DIR / 'kv_cache_benchmark.png'}")


if __name__ == "__main__":
    main()
