from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM, get_batch


def encode_corpus_fast(corpus_path: Path) -> tuple[np.ndarray, int]:
    """优先使用 tiktoken 极速编码；若未安装则回退至 UTF-8 字节级编码"""
    text = corpus_path.read_text(encoding="utf-8")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode_ordinary(text), dtype=np.int32)
        vocab_size = enc.n_vocab
        print(f"[Tokenizer] 使用 tiktoken (GPT-2) 编码完成，总 Token 数: {len(tokens):,}")
    except ImportError:
        tokens = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        vocab_size = 256
        print(f"[Tokenizer] 使用 Byte-level 编码完成，总 Token 数: {len(tokens):,}")
    return tokens, vocab_size


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus_path = Path("tests/fixtures/tinystories_sample_5M.txt")
    assert corpus_path.exists(), f"找不到内置数据集: {corpus_path}"

    tokens, vocab_size = encode_corpus_fast(corpus_path)

    # 轻量级模型配置（适配单张消费级显卡或笔记本，几分钟内完成收敛）
    config = dict(
        vocab_size=vocab_size,
        context_length=1024,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=768,
        rope_theta=10000.0,
    )
    model = TransformerLM(**config, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 初始化完成 | 设备: {device} | 参数量: {num_params / 1e6:.2f}M")

    # 按要求直接调用 PyTorch 官方优化器、余弦退火调度器与交叉熵接口
    max_steps = 300
    batch_size = 8
    grad_accum = 2
    seq_len = 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    losses: list[float] = []
    model.train()
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(tokens, batch_size=batch_size, context_length=seq_len, device=device)

            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)  # (B, S, vocab_size)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) / grad_accum

            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(step_loss)
        if step % 25 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(
                f"Step {step:03d}/{max_steps} | Loss: {step_loss:.4f} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
            )

    # 保存权重与训练收敛曲线
    Path("checkpoints").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    torch.save({"config": config, "model_state_dict": model.state_dict()}, "checkpoints/model.pt")

    plt.figure(figsize=(7, 4.2), dpi=160)
    plt.plot(range(1, max_steps + 1), losses, color="#1f77b4", linewidth=1.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("TinyStories Pre-training Loss Curve (Llama-style Transformer)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("assets/training_loss.png")
    print("[Done] 权重已保存至 checkpoints/model.pt，损失曲线已保存至 assets/training_loss.png")


if __name__ == "__main__":
    main()