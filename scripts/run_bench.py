"""KV Cache 基准入口：等价性断言 + 多长度延迟/显存测量 + CSV 落盘。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

import torch

from kvlm.bench import benchmark, check_equivalence, format_markdown_table, load_model

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def gpu_state() -> str:
    """记录测量时的 GPU 时钟/温度，便于解释跨次运行的吞吐波动。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KV-cache decoding")
    parser.add_argument("--config", type=Path, default=Path("configs/bench.toml"))
    args = parser.parse_args()

    with args.config.open("rb") as f:
        cfg = tomllib.load(f)

    torch.manual_seed(cfg["run"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[cfg["model"].get("dtype", "float16")] if device == "cuda" else torch.float32

    model, config = load_model(
        Path(cfg["model"]["checkpoint"]), device, dtype, int(cfg["model"].get("context_pad", 1088))
    )

    bench_cfg = cfg["bench"]
    prompt_len = int(bench_cfg["prompt_len"])
    check_tokens = int(bench_cfg.get("check_tokens", 64))

    prompt = torch.randint(0, config["vocab_size"], (1, prompt_len), device=device)
    passed = check_equivalence(model, prompt, check_tokens)
    print(f"[Check] 无缓存 vs KV Cache {check_tokens} token 贪心解码一致: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit("KV Cache 与全量重算输出不一致，请检查 RoPE start_pos")

    rows, base_alloc_mb = benchmark(
        model,
        config,
        device,
        dtype,
        prompt_len=prompt_len,
        lengths=[int(x) for x in bench_cfg["lengths"]],
        warmup=int(bench_cfg.get("warmup", 3)),
        repeat=int(bench_cfg.get("repeat", 3)),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"bench_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "run_id": run_id,
        "config": cfg,
        "device": device,
        "dtype": str(dtype),
        "base_alloc_mb": base_alloc_mb,
        "equivalence_pass": passed,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "gpu_state": gpu_state(),
    }
    meta_path = out_dir / f"bench_{run_id}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(format_markdown_table(rows))
    print(f"[Done] CSV   -> {csv_path}")
    print(f"[Done] meta  -> {meta_path}")


if __name__ == "__main__":
    main()
