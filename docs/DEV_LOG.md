# CS336 Assignment 1 迁移与修复报告（2026-10-28 夜间自动执行）

## 1. 环境

- `uv` 0.13.6 已安装到 `~/.local/bin/uv`，可直接在终端使用。
- 全局配置 `~/.config/uv/uv.toml` 已指向清华源：`index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"`。
- 项目环境：仓库根目录 `.venv`（约 4.8GB），Python 3.13.18。
- torch 2.11.0+cu130，`torch.cuda.is_available() == True`，RTX 3050 Ti Laptop（sm_86），fp16 matmul 冒烟测试通过。
- `matplotlib` 3.11.2 已通过 `uv add` 写入 `pyproject.toml` 与 `uv.lock`。
- 注意：`uv.lock` 现已整体指向清华源（625 处 TUNA URL，0 处 pypi.org）。换机器或交作业前建议用官方 PyPI 重新执行一次 `uv lock`。

## 2. 基线（修复前）

命令：`uv run pytest --timeout=300 -q`

结果：**45 passed, 2 failed, 1 xfailed，35.32s**

- `FAILED tests/test_tokenizer.py::test_overlapping_special_tokens` — `assert 3 == 1`
- `FAILED tests/test_train_bpe.py::test_train_bpe_speed` — 实测 **1.64s**，阈值 1.5s
- 其余全绿：模型快照比对无数值漂移；Linux-only 的 `test_encode_iterable_memory_usage` 首次在 Linux 执行即通过；`test_encode_memory_usage` 为官方预期 xfail。

## 3. 修复内容

### 3.1 特殊 Token 最长匹配（tokenizer.py）

- 位置：`src/kvlm/tokenizer.py` 中 `encode()` 的特殊 token 切分处。
- 原因：`re.split` 的交替分支按列表书写顺序匹配，短 token 抢先命中，导致 `"<|endoftext|><|endoftext|>"` 被拆成两个单 token。
- 改动：构造 pattern 前按长度降序排序特殊 token。
- 效果：`test_overlapping_special_tokens` 通过，其余 tokenizer 测试全部保持通过。

### 3.2 BPE 增量统计（bpe.py）

- 位置：`src/kvlm/bpe.py` 的 `train_bpe()`。
- 原因：原实现每一轮都全量重扫所有 pre-token 并重建元组，420+ 次合并耗时 1.64s。
- 改动：改为维护 `pair -> 频次`、`pair -> 包含它的 pre-token 集合` 两套索引，外加惰性删除大顶堆；堆项用 `_DescPair` 包装，保证频次相同时仍选字节序最大的 pair（与原实现平局规则一致）。每次合并只更新真正受影响的 pre-token。
- 效果：
  - `test_train_bpe_speed`：1.64s → **0.63s**
  - `test_train_bpe`：与参考 merges 完全一致（逐条相等）
  - `test_train_bpe_special_tokens`（5MB 语料）：9.34s → 2.58s，快照比对通过

## 4. 回归（修复后）

命令：`uv run pytest --timeout=300 -q`

结果：**47 passed, 1 xfailed，18.96s**

- 唯一 xfail 是官方的 `test_encode_memory_usage`（预期失败）。
- 全程无新增失败。

## 5. 未执行 / 待你决定

1. `train.py` 未运行：4GB 显存下 `batch_size=16` + 50257 词表有 OOM 风险，建议先降到 4–8 或加梯度累积后再跑。
2. `profile_kvcache.py` 未运行：`torch.equal` 严格相等断言在 fp16 下可能因数值尾差翻转 argmax，需实测。
3. `Tokenizer.from_files` 对 JSON 格式 merges 的解析仍可疑（测试未覆盖该路径），需要时再单独验证。
4. `ruff` 报 5 个历史风格告警（UP015/UP035/UP028），均为本次改动之前就存在的代码，未处理。
5. `uv.lock` 是否还原为官方 PyPI 地址，需要你确认。

## 6. 备份

原始文件已备份到项目根目录：

- `.backup_before_fix/bpe.py`
- `.backup_before_fix/tokenizer.py`

恢复方式：把这两个文件覆盖回 `src/kvlm/`。

## 7. 常用命令

```bash
cd /home/luy/Coding/cs336-2026
uv run pytest --timeout=300
uv run python scripts/run_train.py --config configs/train_50m.toml
uv run python scripts/run_bench.py --config configs/bench.toml
```

---

机器已按你的要求在本报告写入后自动关机。
