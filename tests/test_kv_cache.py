"""KV Cache 专项测试：容器语义、显存公式、缓存与全量重算的一致性。"""

from __future__ import annotations

import torch

from kvlm.kv_cache import KVCache
from kvlm.model import TransformerLM


def test_cache_growth_and_crop() -> None:
    cache = KVCache(num_layers=2)
    assert cache.get_seq_len() == 0

    k = torch.zeros(1, 2, 4, 8)
    v = torch.ones(1, 2, 4, 8)
    cache.update(0, k, v)
    cache.update(1, k, v)
    assert cache.get_seq_len(0) == 4
    assert cache.get_seq_len(1) == 4

    cache.update(0, k[:, :, :1], v[:, :, :1])
    assert cache.get_seq_len(0) == 5
    assert cache.get_seq_len(1) == 4

    cache.crop(3)
    assert cache.get_seq_len(0) == 3
    assert cache.get_seq_len(1) == 3
    assert cache.cache[0][0].shape[-2] == 3


def test_memory_bytes_matches_theory() -> None:
    layers, batch, heads, seq, d_head = 3, 1, 2, 6, 4
    cache = KVCache(num_layers=layers)
    block = torch.zeros(batch, heads, seq, d_head, dtype=torch.float32)
    for i in range(layers):
        cache.update(i, block, block)

    expected = layers * 2 * batch * heads * seq * d_head * 4
    assert cache.memory_bytes() == expected


def test_cached_decode_matches_full_recompute() -> None:
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=64,
        context_length=32,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        device="cpu",
    )
    model.eval()
    prompt = torch.randint(0, 64, (1, 5))

    with torch.no_grad():
        out_full = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=False)
        out_cached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=True)

    assert torch.equal(out_full, out_cached)
