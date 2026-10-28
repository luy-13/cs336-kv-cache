"""KV Cache 状态容器。

把逐层的历史 K/V 张量从模型主体中独立出来，便于单测与基准脚本复用。
"""

from __future__ import annotations

import torch
from torch import Tensor


class KVCache:
    """逐层保存历史 Key/Value，把单步解码的注意力计算从 O(T^2) 降到 O(T)。

    每层存一个 ``(k, v)``，形状均为 ``(B, num_heads, seq_len, d_head)``。
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.cache: list[tuple[Tensor, Tensor] | None] = [None] * num_layers

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """返回当前已缓存的历史长度，即下一步的 start_pos。"""
        if self.cache[layer_idx] is None:
            return 0
        return self.cache[layer_idx][0].shape[-2]

    def update(self, layer_idx: int, k_val: Tensor, v_val: Tensor) -> tuple[Tensor, Tensor]:
        """把当前步的 k/v 沿序列维拼接到历史缓存。

        Prefill 阶段输入形状为 ``(B, H, S, d_head)``，Decode 阶段为 ``(B, H, 1, d_head)``。
        """
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (k_val, v_val)
        else:
            prev_k, prev_v = self.cache[layer_idx]
            new_k = torch.cat([prev_k, k_val], dim=-2)
            new_v = torch.cat([prev_v, v_val], dim=-2)
            self.cache[layer_idx] = (new_k, new_v)
        return self.cache[layer_idx]

    def crop(self, max_seq_len: int) -> None:
        """截断缓存尾部到指定长度，用于采样分支回滚。"""
        for i in range(self.num_layers):
            if self.cache[i] is not None:
                k, v = self.cache[i]
                self.cache[i] = (k[..., :max_seq_len, :], v[..., :max_seq_len, :])

    def memory_bytes(self) -> int:
        """按 numel x element_size 统计当前缓存的张量占用。"""
        total_bytes = 0
        for item in self.cache:
            if item is not None:
                k, v = item
                total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total_bytes
