"""语料编码工具：把文本转成 int32 token 数组。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GPT2_VOCAB_SIZE = 50257


def encode_corpus(corpus_path: Path) -> tuple[np.ndarray, int]:
    """用 tiktoken GPT-2 编码语料；未安装时回退到 UTF-8 字节级编码。"""
    text = corpus_path.read_text(encoding="utf-8")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode_ordinary(text), dtype=np.int32)
        vocab_size = enc.n_vocab
        print(f"[Tokenizer] {corpus_path.name}: tiktoken (GPT-2)，Token 数 {len(tokens):,}")
    except ImportError:
        tokens = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        vocab_size = 256
        print(f"[Tokenizer] {corpus_path.name}: byte-level，Token 数 {len(tokens):,}")
    return tokens, vocab_size
