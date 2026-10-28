from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch

from cs336_basics.model import TransformerLM


def sync_device(device: str) -> None:
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
    """理论公式: 2 (K和V) * num_layers * B * num_heads * S * d_head * bytes"""
    total_bytes = 2 * num_layers * batch_size * num_heads * seq_len * d_head * bytes_per_elem
    return total_bytes / (1024**2)


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    ckpt_path = Path("checkpoints/model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        config["context_length"] = max(config.get("context_length", 1024), 1088)
        model = TransformerLM(**config, device=device, dtype=dtype)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt  # 释放 fp32 权重副本，避免污染后续显存基线
    else:
        config = dict(
            vocab_size=50257,
            context_length=1088,
            d_model=256,
            num_layers=6,
            num_heads=8,
            d_ff=768,
            rope_theta=10000.0,
        )
        model = TransformerLM(**config, device=device, dtype=dtype)

    model.eval()
    prompt_len = 16
    prompt = torch.randint(0, config["vocab_size"], (1, prompt_len), device=device)

    # ==========================================================================
    # 1. 数学等价性断言校验 (Sanity Check)
    # ==========================================================================
    print("[Step 1] 正在校验 use_cache=False 与 use_cache=True 的贪心解码输出一致性...")
    out_no_cache = model.generate(prompt, max_new_tokens=64, temperature=0.0, use_cache=False)
    out_with_cache = model.generate(prompt, max_new_tokens=64, temperature=0.0, use_cache=True)
    assert torch.equal(out_no_cache, out_with_cache), "KV Cache 输出与全量重算不一致！请检查 RoPE start_pos"
    print("  -> [PASS] 64 Tokens 贪心解码序列 100% 严格一致！")

    # ==========================================================================
    # 2. 多序列长度延迟、吞吐量与 KV Cache 显存剖析
    # ==========================================================================
    gen_lengths = [64, 128, 256, 512, 1024]
    results = []

    # 预热 GPU
    for _ in range(3):
        _ = model.generate(prompt, max_new_tokens=16, use_cache=True)
    sync_device(device)
    if device == "cuda":
        print(f"  -> 权重与 prompt 基线占用: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    print("\n[Step 2] 开始运行序列长度扩展 Benchmark...")
    for L in gen_lengths:
        # 测无缓存模式 (O(L^2) 单步重算)
        sync_device(device)
        t0 = time.perf_counter()
        _ = model.generate(prompt, max_new_tokens=L, temperature=0.0, use_cache=False)
        sync_device(device)
        no_cache_ms = (time.perf_counter() - t0) * 1000.0
        no_cache_tps = L / (no_cache_ms / 1000.0)

        # 测 KV Cache 增量模式 (O(L) 线性复杂度)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # 清空缓存池，隔离上一个 L 留下的碎片
            base_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
            base_reserved_mb = torch.cuda.memory_reserved() / (1024**2)
            torch.cuda.reset_peak_memory_stats()  # 峰值计数器从当前水位重新起算
        sync_device(device)
        t0 = time.perf_counter()
        _, kv_cache = model.generate(
            prompt, max_new_tokens=L, temperature=0.0, use_cache=True, return_cache=True
        )
        sync_device(device)
        cache_ms = (time.perf_counter() - t0) * 1000.0
        if device == "cuda":
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            steady_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        else:
            base_alloc_mb = base_reserved_mb = peak_alloc_mb = peak_reserved_mb = steady_alloc_mb = 0.0
        cache_tps = L / (cache_ms / 1000.0)
        speedup = no_cache_ms / cache_ms

        total_seq = prompt_len + L
        measured_kv_mb = kv_cache.memory_bytes() / (1024**2)
        del kv_cache  # 释放引用，避免上一个 L 的缓存污染下一轮的基线
        bytes_per_elem = 2 if dtype == torch.float16 else 4
        theory_kv_mb = theoretical_kv_cache_mb(
            batch_size=1,
            seq_len=total_seq,
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_head=config["d_model"] // config["num_heads"],
            bytes_per_elem=bytes_per_elem,
        )

        results.append(
            dict(
                L=L,
                no_cache_ms=no_cache_ms,
                cache_ms=cache_ms,
                no_cache_tps=no_cache_tps,
                cache_tps=cache_tps,
                speedup=speedup,
                measured_kv_mb=measured_kv_mb,
                theory_kv_mb=theory_kv_mb,
                base_alloc_mb=base_alloc_mb,
                peak_alloc_mb=peak_alloc_mb,
                peak_reserved_mb=peak_reserved_mb,
                peak_kv_delta_mb=peak_alloc_mb - base_alloc_mb,
                peak_reserved_delta_mb=peak_reserved_mb - base_reserved_mb,
                steady_kv_delta_mb=steady_alloc_mb - base_alloc_mb,
            )
        )

    # 打印 Markdown 实验表格（可直接粘贴进 README 与简历）
    print("\n### KV Cache 推理加速与显存剖析实验表\n")
    print(
        "| 生成长度 $L$ | 无缓存耗时 (ms) | KV Cache 耗时 (ms) | 无缓存吞吐 (tok/s) "
        "| KV Cache 吞吐 (tok/s) | 加速比 | 张量口径 / 理论 (MB) "
        "| 分配器峰值增量 (MB) |"
    )
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(
            f"| {r['L']} | {r['no_cache_ms']:.1f} | {r['cache_ms']:.1f} "
            f"| {r['no_cache_tps']:.1f} | {r['cache_tps']:.1f} "
            f"| **{r['speedup']:.2f}x** | {r['measured_kv_mb']:.2f} / {r['theory_kv_mb']:.2f} "
            f"| {r['peak_kv_delta_mb']:.2f} |"
        )

    # 绘制双子图
    Path("assets").mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)

    Ls = [r["L"] for r in results]
    ax1.plot(Ls, [r["no_cache_ms"] for r in results], "o--", color="#d62728", label="No Cache (Recompute)")
    ax1.plot(Ls, [r["cache_ms"] for r in results], "s-", color="#1f77b4", label="With KV Cache (Incremental)")
    ax1.set_xlabel("Generated Sequence Length (Tokens)")
    ax1.set_ylabel("End-to-End Latency (ms)")
    ax1.set_title("Decoding Latency: O(L³) vs. O(L²)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(Ls, [r["theory_kv_mb"] for r in results], "k--", label="Theoretical Formula")
    ax2.plot(Ls, [r["measured_kv_mb"] for r in results], "o-", color="#2ca02c", label="Measured KV Tensor Memory")
    ax2.plot(Ls, [r["peak_kv_delta_mb"] for r in results], "^-", color="#d62728", label="Allocator Peak Delta")
    ax2.set_xlabel("Generated Sequence Length (Tokens)")
    ax2.set_ylabel("KV Cache Footprint (MB)")
    ax2.set_title("KV Cache Memory: Theory vs. Tensor vs. Allocator Peak")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("assets/kv_cache_benchmark.png")
    print("\n[Done] 高清对比折线图已导出至 assets/kv_cache_benchmark.png")


if __name__ == "__main__":
    main()