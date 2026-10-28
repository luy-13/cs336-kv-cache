"""KV Cache 基准测量：等价性断言、延迟/吞吐、显存对账。

每个长度重复多次取中位数，降低 GPU 时钟抖动与热降频带来的噪声。
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model import TransformerLM


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def theoretical_kv_cache_mb(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    d_head: int,
    bytes_per_elem: int,
) -> float:
    """2 (K,V) x layers x B x heads x S x d_head x bytes。"""
    total_bytes = 2 * num_layers * batch_size * num_heads * seq_len * d_head * bytes_per_elem
    return total_bytes / (1024**2)


def load_model(checkpoint: Path, device: str, dtype: torch.dtype, context_pad: int) -> tuple[TransformerLM, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    config["context_length"] = max(config.get("context_length", 1024), context_pad)
    model = TransformerLM(**config, device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt  # 释放 fp32 权重副本，避免污染显存基线
    model.eval()
    return model, config


def check_equivalence(model: TransformerLM, prompt: Tensor, new_tokens: int) -> bool:
    """校验无缓存重算与 KV Cache 增量解码的输出逐 token 完全一致。"""
    with torch.no_grad():
        out_no_cache = model.generate(prompt, max_new_tokens=new_tokens, temperature=0.0, use_cache=False)
        out_with_cache = model.generate(prompt, max_new_tokens=new_tokens, temperature=0.0, use_cache=True)
    return bool(torch.equal(out_no_cache, out_with_cache))


def benchmark(
    model: TransformerLM,
    config: dict[str, Any],
    device: str,
    dtype: torch.dtype,
    prompt_len: int,
    lengths: list[int],
    warmup: int = 3,
    repeat: int = 3,
) -> tuple[list[dict[str, Any]], float]:
    """对每个生成长度测量无缓存/缓存两种模式的延迟、吞吐与峰值显存。"""
    prompt = torch.randint(0, config["vocab_size"], (1, prompt_len), device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model.generate(prompt, max_new_tokens=16, use_cache=True)
    sync(device)

    base_alloc_mb = 0.0
    if device == "cuda":
        base_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"[Bench] 权重与 prompt 基线占用 {base_alloc_mb:.1f} MB")

    rows: list[dict[str, Any]] = []
    bytes_per_elem = 2 if dtype == torch.float16 else 4

    with torch.no_grad():
        for L in lengths:
            no_cache_ms_samples: list[float] = []
            cache_ms_samples: list[float] = []
            peak_samples: list[float] = []
            tensor_mb = 0.0

            for _ in range(repeat):
                sync(device)
                t0 = time.perf_counter()
                model.generate(prompt, max_new_tokens=L, temperature=0.0, use_cache=False)
                sync(device)
                no_cache_ms_samples.append((time.perf_counter() - t0) * 1000.0)

                if device == "cuda":
                    torch.cuda.empty_cache()  # 隔离上一个长度的碎片
                    base_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
                    torch.cuda.reset_peak_memory_stats()

                sync(device)
                t0 = time.perf_counter()
                _, kv_cache = model.generate(
                    prompt, max_new_tokens=L, temperature=0.0, use_cache=True, return_cache=True
                )
                sync(device)
                cache_ms_samples.append((time.perf_counter() - t0) * 1000.0)

                tensor_mb = kv_cache.memory_bytes() / (1024**2)
                if device == "cuda":
                    peak_samples.append(torch.cuda.max_memory_allocated() / (1024**2) - base_alloc_mb)
                    torch.cuda.reset_peak_memory_stats()
                del kv_cache

            no_cache_ms = statistics.median(no_cache_ms_samples)
            cache_ms = statistics.median(cache_ms_samples)
            peak_kv_delta_mb = statistics.median(peak_samples) if peak_samples else 0.0
            rows.append(
                {
                    "L": L,
                    "no_cache_ms": no_cache_ms,
                    "cache_ms": cache_ms,
                    "no_cache_tps": L / (no_cache_ms / 1000.0),
                    "cache_tps": L / (cache_ms / 1000.0),
                    "speedup": no_cache_ms / cache_ms,
                    "tensor_kv_mb": tensor_mb,
                    "theory_kv_mb": theoretical_kv_cache_mb(
                        batch_size=1,
                        seq_len=prompt_len + L,
                        num_layers=config["num_layers"],
                        num_heads=config["num_heads"],
                        d_head=config["d_model"] // config["num_heads"],
                        bytes_per_elem=bytes_per_elem,
                    ),
                    "peak_kv_delta_mb": peak_kv_delta_mb,
                }
            )
            print(
                f"  L={L:4d} | no-cache {no_cache_ms:8.1f} ms | cache {cache_ms:8.1f} ms "
                f"| speedup {rows[-1]['speedup']:.2f}x | peak +{peak_kv_delta_mb:.2f} MB (median of {repeat})"
            )
    return rows, base_alloc_mb


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| 生成长度 L | 无缓存 (ms) | Cache (ms) | 无缓存 (tok/s) | Cache (tok/s) | 加速比 "
        "| 张量 / 理论 (MB) | 分配器峰值增量 (MB) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {r['L']} | {r['no_cache_ms']:.1f} | {r['cache_ms']:.1f} "
        f"| {r['no_cache_tps']:.1f} | {r['cache_tps']:.1f} | {r['speedup']:.2f}x "
        f"| {r['tensor_kv_mb']:.2f} / {r['theory_kv_mb']:.2f} | {r['peak_kv_delta_mb']:.2f} |\n"
        for r in rows
    )
    return header + body
