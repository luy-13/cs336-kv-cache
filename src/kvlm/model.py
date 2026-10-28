from __future__ import annotations

import math
import os
from typing import BinaryIO, IO
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor

from .kv_cache import KVCache


# ==============================================================================
# 1. 基础组件：Linear, Embedding, RMSNorm, SiLU, SwiGLU (CS336 §3.1 - §3.3)
# ==============================================================================

class Linear(nn.Module):
    """无偏置线性层 (CS336 §3.1): y = x @ W^T"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight 形状: (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_in), self.weight: (d_out, d_in) -> out: (..., d_out)
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """词嵌入查找表 (CS336 §3.1)"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # weight 形状: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (B, S) -> out: (B, S, d_model)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """均方根层归一化 (CS336 §3.2): RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # weight (gain) 形状: (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # 为防止 FP16/BF16 下平方求和溢出，先转为 float32 计算均方根
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)  # (..., d_model)
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        x_norm = (x_fp32 * rms).to(in_dtype)  # (..., d_model)
        return x_norm * self.weight  # (..., d_model)


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish) 激活函数: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """门控前馈网络 (CS336 §3.3): FFN(x) = W2(SiLU(W1(x)) * W3(x))"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # (d_ff, d_model)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # (d_model, d_ff)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # (d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_model)
        # w1(x): (..., d_ff), w3(x): (..., d_ff)
        gated = silu(self.w1(x)) * self.w3(x)  # (..., d_ff)
        return self.w2(gated)  # (..., d_model)


# ==============================================================================
# 2. 旋转位置编码 RoPE 与 KV Cache 状态容器 (CS336 §3.4 & 加分项一)
# ==============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 RoPE (CS336 §3.4)
    将最后一维 d_k 按相邻两维 (2k, 2k+1) 配对进行二维平面旋转。
    支持传入任意 token_positions（在 KV Cache Decode 阶段传入 start_pos 偏移位置）。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率: theta_k = theta^(-2k / d_k), k = 0, 1, ..., d_k/2 - 1
        k_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # (d_k // 2,)
        inv_freq = 1.0 / (self.theta ** (k_indices / d_k))  # (d_k // 2,)

        # 预计算 [0, max_seq_len) 所有位置的 cos 和 sin 缓存表
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len,)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k // 2)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Args:
            x: 输入 Query 或 Key，形状为 (..., seq_len, d_k)，例如 (B, H, S, d_k) 或 (B, S, d_k)
            token_positions: 位置索引张量，形状为 (..., seq_len)，例如 (B, S) 或 (S,)
        Returns:
            旋转编码后的张量，形状与 x 完全一致 (..., seq_len, d_k)
        """
        in_dtype = x.dtype
        # 根据 token_positions 从预计算表中取出对应位置的 cos 和 sin
        # cos, sin 形状: (*token_positions.shape, d_k // 2)
        cos = self.cos_cached[token_positions].to(torch.float32)
        sin = self.sin_cached[token_positions].to(torch.float32)

        # 若 x 带有多头维度 (B, H, S, d_k)，而 token_positions 为 (B, S) 或 (S,)，自动对齐广播维度
        while cos.ndim < x.ndim:
            # 若 token_positions 是 (B, S) 且 x 是 (B, H, S, d_k)，在 head 维 (dim=-3) 插入 1
            insert_dim = -3 if cos.ndim >= 2 else 0
            cos = cos.unsqueeze(insert_dim)
            sin = sin.unsqueeze(insert_dim)

        # 将最后一维 d_k 拆分为相邻偶数项 x_even (2k) 与奇数项 x_odd (2k+1)
        x_fp32 = x.to(torch.float32)
        x_even = x_fp32[..., 0::2]  # (..., seq_len, d_k // 2)
        x_odd = x_fp32[..., 1::2]   # (..., seq_len, d_k // 2)

        # 二维旋转矩阵乘法:
        # [x'_even] = [cos  -sin] [x_even]
        # [x'_odd ]   [sin   cos] [x_odd ]
        rot_even = x_even * cos - x_odd * sin  # (..., seq_len, d_k // 2)
        rot_odd = x_even * sin + x_odd * cos   # (..., seq_len, d_k // 2)

        # 重新交织拼回 (..., seq_len, d_k)
        out = torch.stack((rot_even, rot_odd), dim=-1).flatten(start_dim=-2)
        return out.to(in_dtype)


# ==============================================================================
# 3. 注意力机制与 Transformer 主干 (CS336 §3.4 - §3.6)
# ==============================================================================

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """数值稳定的手写 Softmax: 减去目标维度最大值防止 exp 溢出"""
    x_fp32 = x.to(torch.float32)
    x_max = torch.amax(x_fp32, dim=dim, keepdim=True)
    exp_x = torch.exp(x_fp32 - x_max)
    out = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return out.to(x.dtype)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    缩放点积注意力 (CS336 §3.4):
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
    Args:
        Q: (..., S_q, d_k)
        K: (..., S_k, d_k)
        V: (..., S_k, d_v)
        mask: 布尔掩码 (..., S_q, S_k)，True 表示允许关注，False 表示屏蔽为 -inf
    Returns:
        out: (..., S_q, d_v)
    """
    d_k = Q.shape[-1]
    # scores: (..., S_q, S_k)
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)  # (..., S_q, S_k)
    return attn_weights @ V  # (..., S_q, d_v)


class MultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力 (CS336 §3.4)
    同时支持：
      1. 标准无位置编码模式（用于通过 test_multihead_self_attention）
      2. 带 RoPE 旋转位置编码模式（用于通过 test_multihead_self_attention_with_rope）
      3. 带 KVCache 的 Prefill / Decode 增量推理模式（加分项一）
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float | None = None,
        use_rope: bool = True,
        layer_idx: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.layer_idx = layer_idx
        self.use_rope = use_rope and (theta is not None)

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        # x 形状: (..., S_q, d_model)，支持任意前导 Batch 维
        *batch_dims, seq_len, _ = x.shape

        # 确定当前序列的起始位置偏移量 start_pos（用于 KV Cache 增量解码时的 RoPE 对齐）
        start_pos = kv_cache.get_seq_len(self.layer_idx) if kv_cache is not None else 0

        # 1. 线性投影并拆分为多头: (..., S_q, d_model) -> (..., num_heads, S_q, d_head)
        q = self.q_proj(x).view(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2, -3)
        k = self.k_proj(x).view(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2, -3)
        v = self.v_proj(x).view(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2, -3)

        # 2. 对 Query 和 Key 施加 RoPE 旋转位置编码（必须在写入 KV Cache 之前旋转！）
        if self.rope is not None:
            if token_positions is None:
                # 构造绝对位置 [start_pos, start_pos + 1, ..., start_pos + seq_len - 1]
                token_positions = torch.arange(
                    start_pos, start_pos + seq_len, device=x.device, dtype=torch.long
                )
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 3. 若启用了 KV Cache，更新并获取完整历史 Key/Value: (..., num_heads, S_kv, d_head)
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        total_seq_len = k.shape[-2]

        # 4. 构造因果掩码 (Causal Mask): True 表示允许关注 (key_pos <= query_pos)
        if seq_len == 1 and kv_cache is not None:
            # 单 Token 增量 Decode 阶段：当前最新 Query 可以关注缓存中的所有历史 Key，无需掩码
            causal_mask = None
        else:
            q_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(-1)  # (S_q, 1)
            k_idx = torch.arange(0, total_seq_len, device=x.device).unsqueeze(0)                 # (1, S_kv)
            causal_mask = q_idx >= k_idx  # (S_q, S_kv)，下三角为 True

        # 5. 缩放点积注意力与多头合并投影
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)  # (..., H, S_q, d_head)
        attn_out = attn_out.transpose(-2, -3).contiguous().view(*batch_dims, seq_len, self.d_model)
        return self.output_proj(attn_out)  # (..., S_q, d_model)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer 层 (CS336 §3.5)"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        layer_idx: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        # 子层 1: Pre-RMSNorm + 因果多头自注意力 + 残差连接
        x = x + self.attn(self.ln1(x), token_positions=token_positions, kv_cache=kv_cache)
        # 子层 2: Pre-RMSNorm + SwiGLU 前馈网络 + 残差连接
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    完整的 Decoder-only Transformer 语言模型 (CS336 §3.6)
    状态字典键名严格对齐 CS336 官方测试：
      - token_embeddings.weight
      - layers.{i}.ln1.weight, layers.{i}.attn.{q,k,v,output}_proj.weight
      - layers.{i}.ln2.weight, layers.{i}.ffn.{w1,w2,w3}.weight
      - ln_final.weight
      - lm_head.weight
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    layer_idx=i,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Tensor,
        token_positions: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        # in_indices: Prefill 阶段为 (B, S)，Decode 阶段为 (B, 1)
        x = self.token_embeddings(in_indices)  # (B, S_q, d_model)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions, kv_cache=kv_cache)
        x = self.ln_final(x)  # (B, S_q, d_model)
        logits = self.lm_head(x)  # (B, S_q, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_cache: bool = True,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache | None]:
        """
        支持无缓存全量重算 (use_cache=False) 与 KV Cache 增量解码 (use_cache=True) 的自回归生成管线。
        """
        self.eval()
        generated = prompt_ids.clone()
        kv_cache = KVCache(self.num_layers) if use_cache else None

        for step in range(max_new_tokens):
            if use_cache:
                # Step 0 (Prefill): 喂入完整 Prompt (B, S_prompt)
                # Step >= 1 (Decode): 每次仅喂入最新生成的 1 个 Token (B, 1)
                curr_input = generated if step == 0 else generated[:, -1:]
                logits = self.forward(curr_input, kv_cache=kv_cache)
            else:
                # 无缓存模式: 每一步都截断上下文窗口并重算完整序列
                curr_input = generated[:, -self.context_length :]
                logits = self.forward(curr_input, kv_cache=None)

            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                probs = softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

        if return_cache:
            return generated, kv_cache
        return generated


# ==============================================================================
# 4. 训练辅助组件（用于完整通过 CS336 其余 pytest 测试项）
# ==============================================================================

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """数值稳定的交叉熵损失 (CS336 §4.1)"""
    logits_fp32 = inputs.to(torch.float32)
    max_logits = torch.amax(logits_fp32, dim=-1, keepdim=True)
    shifted = logits_fp32 - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """全局梯度 L2 范数裁剪 (CS336 §4.3)"""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(torch.sum(g.detach().to(torch.float32) ** 2) for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


class AdamW(torch.optim.Optimizer):
    """解耦权重衰减的 AdamW 优化器 (CS336 §4.2)"""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                p.addcdiv_(m, torch.sqrt(v) + eps, value=-alpha_t)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带线性预热的余弦退火学习率调度 (CS336 §4.3)"""
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """随机采样语言模型训练批次 (CS336 §5.1)"""
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    x_np = np.stack([dataset[i : i + context_length] for i in starts])
    y_np = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])
    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return x, y


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """保存训练状态字典与迭代步数 (CS336 §5.2)"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """加载训练状态字典并返回迭代步数 (CS336 §5.2)"""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["iteration"])