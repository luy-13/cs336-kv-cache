# 训练与 KV Cache 基准结果（2026-10-28 01:45）

## 1. 训练（batch 8 + 梯度累积 2，等效 batch 16）

| 项目 | 结果 |
|---|---|
| 命令 | `uv run python train.py` |
| 语料 | `tests/fixtures/tinystories_sample_5M.txt`（tiktoken GPT-2，1,328,129 tokens） |
| 模型 | 30.85M 参数，6 层 / d_model 256 / 8 头 / d_ff 768 / ctx 1024 |
| 步数 | 300（等效 batch 16 × seq 256 ≈ 123 万 token，约 1 轮） |
| 耗时 | 57.9 s（RTX 3050 Ti 4GB，bf16 autocast） |
| Loss | **10.8349 → 3.0784** |
| 产物 | `checkpoints/model.pt`（118MB）、`assets/training_loss.png`（1120×672） |

Loss 采样：step1 10.83 → step25 5.76 → step50 4.67 → step100 3.72 → step200 3.15 → step300 3.08。

## 2. KV Cache 基准

一致性校验：`use_cache=False` 与 `use_cache=True` 的 64-token 贪心解码序列**完全一致（PASS）**。

| 生成长度 L | 无缓存 (ms) | KV Cache (ms) | 无缓存吞吐 (tok/s) | Cache 吞吐 (tok/s) | 加速比 | KV 显存实测/理论 (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 317.5 | 267.5 | 201.6 | 239.3 | 1.19x | 0.46 / 0.47 |
| 128 | 600.5 | 523.4 | 213.1 | 244.6 | 1.15x | 0.84 / 0.84 |
| 256 | 1190.5 | 1033.9 | 215.0 | 247.6 | 1.15x | 1.59 / 1.59 |
| 512 | 2609.7 | 2094.0 | 196.2 | 244.5 | 1.25x | 3.09 / 3.09 |
| 1024 | 10595.0 | 4482.5 | 96.6 | 228.4 | **2.36x** | 6.09 / 6.09 |

产物：`assets/kv_cache_benchmark.png`（1840×704，双子图）。

## 3. 结果解读与注意点

- 加速比在 L=1024 才明显（2.36x），短序列只有 1.15–1.25x。原因：`KVCache.update` 每步用 `torch.cat` 复制整段历史，单步开销被放大的同时，模型很小、kernel 启动开销占比高。这会在报告中被导师追问，建议在 README 里如实说明，或改为预分配缓冲区。
- “KV 显存实测”仍是 `memory_bytes()` 的张量口径（按 numel 计算），与理论公式同源；**没有用 `torch.cuda.max_memory_allocated()` 测设备峰值**。审查报告里已列为 P0。
- 本次未应用审查报告中的其他补丁（generate 越界保护、真实峰值显存测量、README / STUDY_GUIDE）。

## 4. 复现

```bash
cd /home/luy/Coding/cs336-2026/assignments/assignment1-basics
uv run python train.py
uv run python profile_kvcache.py
```

原始日志已复制到 `logs/train.log`、`logs/profile.log`。

---

机器已按要求在结果写入后自动关机。
