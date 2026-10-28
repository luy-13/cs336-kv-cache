"""训练入口：读取 TOML 配置，训练并落盘 checkpoint 与 metrics。"""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

import torch

from kvlm.train import train


def git_meta() -> dict[str, str | bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Llama-style decoder-only Transformer")
    parser.add_argument("--config", type=Path, default=Path("configs/train_50m.toml"))
    args = parser.parse_args()

    with args.config.open("rb") as f:
        cfg = tomllib.load(f)

    metrics, model = train(cfg)

    checkpoint = Path(cfg["optim"].get("checkpoint", "checkpoints/model.pt"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = model.vocab_size
    torch.save({"config": model_cfg, "model_state_dict": model.state_dict()}, checkpoint)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics.update(
        {
            "run_id": run_id,
            "git": git_meta(),
            "env": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
        }
    )

    out_path = Path("results/metrics") / f"train_{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    final = metrics["final"]
    print(
        f"[Done] train {final['train']:.4f} | val {final['val']:.4f} | "
        f"{metrics['time_s']:.1f}s | {metrics['tokens_per_s']:,.0f} tok/s"
    )
    print(f"[Done] checkpoint -> {checkpoint}")
    print(f"[Done] metrics    -> {out_path}")


if __name__ == "__main__":
    main()
