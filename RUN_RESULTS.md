# 训练与 KV Cache 基准结果（2026-10-28 更新）

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

## 2. KV Cache 基准（含真实峰值显存）

一致性校验：`use_cache=False` 与 `use_cache=True` 的 64-token 贪心解码序列**完全一致（PASS）**。

| 生成长度 L | 无缓存 (ms) | KV Cache (ms) | 无缓存吞吐 (tok/s) | Cache 吞吐 (tok/s) | 加速比 | 张量口径 / 理论 (MB) | 分配器峰值增量 (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 291.2 | 252.4 | 219.8 | 253.6 | 1.15x | 0.46 / 0.47 | 1.73 |
| 128 | 588.6 | 515.2 | 217.5 | 248.4 | 1.14x | 0.84 / 0.84 | 1.73 |
| 256 | 1168.1 | 1036.3 | 219.2 | 247.0 | 1.13x | 1.59 / 1.59 | 1.95 |
| 512 | 2637.0 | 2048.6 | 194.2 | 249.9 | 1.29x | 3.09 / 3.09 | 3.71 |
| 1024 | 10570.1 | 4493.1 | 96.9 | 227.9 | **2.35x** | 6.09 / 6.09 | 7.21 |

### 测量口径

- **张量口径**：`KVCache.memory_bytes()`，按 `numel × element_size` 累加，与理论公式
  `2 × layers × B × heads × d_head × T × bytes` 同源。
- **分配器峰值增量**：每个 L 测前 `torch.cuda.empty_cache()` + `reset_peak_memory_stats()`，
  测后读 `max_memory_allocated()` 并减去基线（权重 + prompt = 67.8 MB）。
  这是设备上真实发生的同时占用峰值，不是把显存跑满。
- **峰值增量 ≈ 张量 × (1 + 1/num_layers)**：`torch.cat` 扩容的瞬间，只有**当前层**同时持有
  新旧两块，其余 5 层已换成新块。实测 7.21 / 6.09 = **1.18×**，与 1 + 1/6 ≈ 1.17 吻合，
  而不是“整体翻倍”。
- `max_memory_reserved()` 受分配器段粒度影响，多数长度下未触发新段（增量为 0），
  因此不纳入表格；它衡量的是缓存池，不是 KV 本身。

产物：`assets/kv_cache_benchmark.png`（双子图：左=延迟，右=理论 / 张量 / 分配器峰值三条曲线）。

## 3. 结果解读

- 加速比在 L=1024 才明显（2.35x），短序列只有 1.13–1.29x。原因：`KVCache.update` 每步用
  `torch.cat` 复制整段历史（单步 O(T) 拷贝、全程 O(T²)），加上模型很小、kernel 启动开销占比高。
  改为预分配缓冲区（`torch.empty(max_len)` + 切片写入）可进一步拉开差距。
- KV 显存随长度线性增长，1024 时 6.09 MB，与理论公式逐点一致；即便算上 cat 的瞬时峰值也只有
  7.21 MB，远不是显存压力来源。
- 真正占显存的是权重（30.85M × 2 bytes ≈ 61.7 MB）与训练时的 logits（B×S×50257）。

## 4. 复现

```bash
cd /home/luy/Coding/cs336-2026/assignments/assignment1-basics
uv run python train.py
uv run python profile_kvcache.py
```

原始日志已复制到 `logs/train.log`、`logs/profile.log`。
