from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from kvlm.model import TransformerLM, get_batch

TRAIN_PATH = Path("data/tinystories_train_50M.txt")
VALID_PATH = Path("data/tinystories_valid_14M.txt")


def encode_corpus_fast(corpus_path: Path) -> tuple[np.ndarray, int]:
    """优先使用 tiktoken 极速编码；若未安装则回退至 UTF-8 字节级编码"""
    text = corpus_path.read_text(encoding="utf-8")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode_ordinary(text), dtype=np.int32)
        vocab_size = enc.n_vocab
        print(f"[Tokenizer] {corpus_path.name}: tiktoken (GPT-2) 编码完成，Token 数: {len(tokens):,}")
    except ImportError:
        tokens = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        vocab_size = 256
        print(f"[Tokenizer] {corpus_path.name}: Byte-level 编码完成，Token 数: {len(tokens):,}")
    return tokens, vocab_size


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert TRAIN_PATH.exists(), f"找不到训练集: {TRAIN_PATH}"
    assert VALID_PATH.exists(), f"找不到验证集: {VALID_PATH}"

    torch.manual_seed(1337)
    np.random.seed(1337)

    train_tokens, vocab_size = encode_corpus_fast(TRAIN_PATH)
    val_tokens, _ = encode_corpus_fast(VALID_PATH)

    # 轻量级模型配置（适配单张消费级显卡或笔记本）
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

    max_steps = 3000
    batch_size = 8
    grad_accum = 2
    seq_len = 256
    val_interval = 200
    tokens_per_step = batch_size * grad_accum * seq_len

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    def evaluate(iters: int = 20) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(iters):
                x, y = get_batch(val_tokens, batch_size=batch_size, context_length=seq_len, device=device)
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                    total += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        model.train()
        return total / iters

    losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    model.train()
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_tokens, batch_size=batch_size, context_length=seq_len, device=device)

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
                f"Step {step:04d}/{max_steps} | Loss: {step_loss:.4f} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
            )

        if step % val_interval == 0 or step == max_steps:
            val_loss = evaluate()
            val_steps.append(step)
            val_losses.append(val_loss)
            print(f"  [Val] step {step:04d} | Val Loss: {val_loss:.4f}")

    elapsed = time.perf_counter() - t0
    tokens_seen = max_steps * tokens_per_step
    print(f"[Train] 完成 {max_steps} 步，耗时 {elapsed:.1f}s，处理 {tokens_seen:,} tokens "
          f"({tokens_seen / elapsed:,.0f} tokens/s)")
    print(f"[Train] Train Loss {losses[0]:.4f} -> {losses[-1]:.4f} | "
          f"Val Loss {val_losses[0]:.4f} -> {val_losses[-1]:.4f}")

    # 保存权重与训练收敛曲线
    Path("checkpoints").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    torch.save({"config": config, "model_state_dict": model.state_dict()}, "checkpoints/model.pt")

    window = 20
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(7.2, 4.2), dpi=160)
    plt.plot(range(1, max_steps + 1), losses, color="#1f77b4", linewidth=0.6, alpha=0.3)
    plt.plot(range(window, max_steps + 1), smooth, color="#1f77b4", linewidth=1.8, label="Train (EMA-20)")
    plt.plot(val_steps, val_losses, "o-", color="#d62728", linewidth=1.5, markersize=3.5, label="Validation")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("TinyStories Pre-training: Train vs. Validation Loss")
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("assets/training_loss.png")
    print("[Done] 权重已保存至 checkpoints/model.pt，损失曲线已保存至 assets/training_loss.png")


if __name__ == "__main__":
    main()
