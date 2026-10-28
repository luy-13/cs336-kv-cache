# 实验报告：50 MB TinyStories 预训练 + KV Cache 推理剖析

> 数据来源：`results/metrics/train_20261028_141228.json` 与 `results/metrics/bench_20261028_142058.csv`。
> 所有数字均可通过 `make train` / `make bench` 复现，图由 `make figures` 从 metrics 重建。

## 1. 训练

| 项目 | 结果 |
|---|---|
| 训练集 | `data/tinystories_train_50M.txt`（50.0 MB，65,077 条故事，12,978,790 tokens） |
| 验证集 | `data/tinystories_valid_14M.txt`（12.9 MB，16,619 条故事，3,335,614 tokens） |
| 数据来源 | Gitee 镜像 `hf-datasets/TinyStories` 流式读取 64 MB 后按 `<|endoftext|>` 切分 |
| 模型 | 30.85M 参数，6 层 / d_model 256 / 8 头 / d_ff 768 / ctx 1024 |
| 配置 | batch 8 × 梯度累积 2 = 16，seq 256，3000 步，AdamW lr 1e-3，余弦退火，seed 1337 |
| 训练量 | 12,288,000 tokens（约 0.95 轮语料） |
| 耗时 / 吞吐 | 716.8 s / 17,142 tokens/s |
| 训练 Loss | 10.8345 → 1.9654 |
| 验证 Loss | 3.4093 → 2.0341（每 200 步在 20 个 batch 上评测） |

验证 Loss 采样：step200 3.4093 → 600 2.6871 → 1000 2.4706 → 1400 2.3245 →
2000 2.1031 → 2400 2.0442 → 3000 2.0341。验证曲线始终在训练曲线上方 0.06–0.10，无明显过拟合。

**关于耗时**：同一配置在不同时段实测 585.1 s 与 716.8 s（17k–23k tok/s），Loss 轨迹完全一致。
差异来自笔记本 GPU 的功耗墙与温度（bench 时记录到 2002 MHz / 62 °C / 33.6 W）。
固定 seed 保证结果可对齐，吞吐数值应按量级理解而非精确值。

## 2. KV Cache 基准

一致性校验：`use_cache=False` 与 `use_cache=True` 的 64-token 贪心解码输出逐 token完全一致（PASS）。
基线（权重 + prompt）：67.8 MB。每个长度重复 3 次取中位数。

| L | 无缓存 (ms) | Cache (ms) | 无缓存 (tok/s) | Cache (tok/s) | 加速比 | 张量 / 理论 (MB) | 分配器峰值增量 (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 331.2 | 285.7 | 193.2 | 224.0 | 1.16x | 0.46 / 0.47 | 1.73 |
| 128 | 656.6 | 574.7 | 194.9 | 222.7 | 1.14x | 0.84 / 0.84 | 1.73 |
| 256 | 1306.2 | 1134.5 | 196.0 | 225.6 | 1.15x | 1.59 / 1.59 | 1.95 |
| 512 | 2990.6 | 2339.5 | 171.2 | 218.8 | 1.28x | 3.09 / 3.09 | 3.71 |
| 1024 | 12767.4 | 5503.2 | 80.2 | 186.1 | **2.32x** | 6.09 / 6.09 | 7.21 |

### 测量口径

- **张量口径**：`KVCache.memory_bytes()`，按 `numel × element_size` 累加，与理论公式
  `2 × layers × B × heads × d_head × T × bytes` 同源。
- **分配器峰值增量**：每次测量前 `empty_cache()` + `reset_peak_memory_stats()`，
  测后读 `max_memory_allocated()` 并减去基线。
- **峰值增量 ≈ 张量 × (1 + 1/num_layers)**：`torch.cat` 扩容瞬间只有当前层同时持有新旧两块，
  其余层已替换完成。实测 7.21 / 6.09 = 1.184，与 1 + 1/6 ≈ 1.167 吻合。
- `max_memory_reserved()` 受分配器段粒度影响，多数长度增量为 0，不纳入表格。

## 3. 解读

- 加速比在 L=1024 才明显（2.32x），短序列只有 1.14–1.16x：模型小、单步计算量低，
  两种模式都被 kernel launch / Python dispatch 的固定开销主导。
- KV 显存线性增长，1024 时 6.09 MiB，算上 cat 瞬时峰值也只有 7.21 MiB，远不是显存压力来源；
  真正的显存开销是 61.7 MB 权重与训练时 `B×S×50257` 的 logits。
- 优化方向：用预分配缓冲区替掉 `torch.cat`，可消除 O(T²) 拷贝，进一步拉高长序列加速比。

## 4. 复现

```bash
uv sync
uv run python scripts/fetch_data.py
uv run python scripts/run_train.py --config configs/train_50m.toml
uv run python scripts/run_bench.py --config configs/bench.toml
uv run python scripts/plot.py
```

日志：`results/logs/train.log`、`results/logs/profile.log`。
