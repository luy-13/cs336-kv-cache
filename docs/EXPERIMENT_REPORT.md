# 训练与 KV Cache 基准结果（2026-10-28 更新）

## 1. 训练（50 MB TinyStories 子集，3000 步）

| 项目 | 结果 |
|---|---|
| 命令 | `uv run python train.py` |
| 训练集 | `data/tinystories_train_50M.txt`（50.0 MB，65,077 条故事，**12,978,790 tokens**） |
| 验证集 | `data/tinystories_valid_14M.txt`（12.9 MB，16,619 条故事，3,335,614 tokens） |
| 数据来源 | Gitee 镜像 `hf-datasets/TinyStories` 流式读取 64 MB 后切分（见 `fetch_data.py`） |
| 模型 | 30.85M 参数，6 层 / d_model 256 / 8 头 / d_ff 768 / ctx 1024 |
| 配置 | batch 8 × 梯度累积 2 = 16，seq 256，3000 步，AdamW lr 1e-3，余弦退火，seed 1337 |
| 训练量 | **12,288,000 tokens**（约 0.95 轮语料） |
| 耗时 | **585.1 s**（≈ 21,001 tokens/s，RTX 3050 Ti 4GB，bf16 autocast） |
| 训练 Loss | **10.8345 → 1.9654** |
| 验证 Loss | **3.4093 → 2.0341**（每 200 步在 20 个 batch 上评测） |
| 产物 | `checkpoints/model.pt`（118 MB）、`assets/training_loss.png`（train EMA + val 双曲线） |

验证 Loss 采样：step200 3.4093 → 600 2.6871 → 1000 2.4706 → 1400 2.3245 →
2000 2.1031 → 2400 2.0442 → 3000 2.0341。验证曲线始终在训练曲线上方约 0.06–0.10，
无明显过拟合。

## 2. KV Cache 基准（同一 checkpoint）

一致性校验：`use_cache=False` 与 `use_cache=True` 的 64-token 贪心解码序列
**逐 token 完全一致（PASS）**。基线（权重 + prompt）：67.8 MB。

| 生成长度 L | 无缓存 (ms) | KV Cache (ms) | 无缓存吞吐 (tok/s) | Cache 吞吐 (tok/s) | 加速比 | 张量口径 / 理论 (MB) | 分配器峰值增量 (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 302.1 | 258.7 | 211.8 | 247.4 | 1.17x | 0.46 / 0.47 | 1.73 |
| 128 | 599.1 | 520.9 | 213.6 | 245.7 | 1.15x | 0.84 / 0.84 | 1.73 |
| 256 | 1176.5 | 1034.6 | 217.6 | 247.4 | 1.14x | 1.59 / 1.59 | 1.95 |
| 512 | 2623.1 | 2065.9 | 195.2 | 247.8 | 1.27x | 3.09 / 3.09 | 3.71 |
| 1024 | 10638.5 | 4583.0 | 96.3 | 223.4 | **2.32x** | 6.09 / 6.09 | 7.21 |

### 测量口径

- **张量口径**：`KVCache.memory_bytes()`，按 `numel × element_size` 累加，与理论公式
  `2 × layers × B × heads × d_head × T × bytes` 同源。
- **分配器峰值增量**：每个 L 测前 `torch.cuda.empty_cache()` + `reset_peak_memory_stats()`，
  测后读 `max_memory_allocated()` 并减去 67.8 MB 基线。这是设备上真实发生的同时占用峰值，
  不是把显存跑满。
- **峰值增量 ≈ 张量 × (1 + 1/num_layers)**：`torch.cat` 扩容的瞬间，只有**当前层**同时持有
  新旧两块，其余 5 层已换成新块。实测 7.21 / 6.09 = **1.18×**，与 1 + 1/6 ≈ 1.17 吻合，
  并非整体翻倍。
- `max_memory_reserved()` 受分配器段粒度影响，多数长度下未触发新段（增量为 0），
  不纳入表格。

产物：`assets/kv_cache_benchmark.png`（左=延迟对数图，右=理论 / 张量 / 分配器峰值三条曲线）。

## 3. 结果解读

- 加速比在 L=1024 才明显（2.32x），短序列只有 1.14–1.17x。原因：`KVCache.update` 每步用
  `torch.cat` 复制整段历史（单步 O(T) 拷贝、全程 O(T²)），加上模型很小、kernel 启动开销占比高。
  改为预分配缓冲区（`torch.empty(max_len)` + 切片写入）可进一步拉开差距。
- KV 显存随长度线性增长，1024 时 6.09 MB，与理论公式逐点一致；算上 cat 的瞬时峰值也只有
  7.21 MB，远不是显存压力来源。
- 真正占显存的是权重（30.85M × 2 bytes ≈ 61.7 MB）与训练时的 logits（B×S×50257）。

## 4. 复现

```bash
cd /home/luy/Coding/cs336-2026/assignments/assignment1-basics
uv run python fetch_data.py     # 拉取 64MB 并切分 train/valid
uv run python train.py          # 3000 步，约 10 分钟
uv run python profile_kvcache.py
```

原始日志已复制到 `logs/train.log`、`logs/profile.log`。
