"""预训练循环：只负责训练本身，不打印表格、不画图。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import encode_corpus
from .model import TransformerLM, get_batch


def _amp_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train(cfg: dict[str, Any]) -> tuple[dict[str, Any], TransformerLM]:
    """按配置训练模型，返回 metrics 与模型。"""
    device = cfg["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["run"].get("seed", 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tokens, vocab_size = encode_corpus(Path(cfg["data"]["train"]))
    val_tokens, _ = encode_corpus(Path(cfg["data"]["valid"]))

    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = vocab_size
    model = TransformerLM(**model_cfg, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {num_params / 1e6:.2f}M 参数 | device={device}")

    opt_cfg = cfg["optim"]
    max_steps = int(opt_cfg["max_steps"])
    batch_size = int(opt_cfg["batch_size"])
    grad_accum = int(opt_cfg["grad_accum"])
    seq_len = int(opt_cfg["seq_len"])
    val_interval = int(opt_cfg.get("val_interval", 0))
    val_batches = int(opt_cfg.get("val_batches", 20))
    tokens_per_step = batch_size * grad_accum * seq_len

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=float(opt_cfg["eta_min"])
    )

    use_amp = device == "cuda"
    amp_dtype = _amp_dtype(device)

    def evaluate() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(val_batches):
                x, y = get_batch(val_tokens, batch_size=batch_size, context_length=seq_len, device=device)
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                    total += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        model.train()
        return total / val_batches

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
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) / grad_accum
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(opt_cfg.get("grad_clip", 1.0)))
        optimizer.step()
        scheduler.step()

        losses.append(step_loss)
        if step % 25 == 0 or step == 1:
            print(f"Step {step:04d}/{max_steps} | Loss {step_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

        if val_interval and (step % val_interval == 0 or step == max_steps):
            val_loss = evaluate()
            val_steps.append(step)
            val_losses.append(val_loss)
            print(f"  [Val] step {step:04d} | Val Loss {val_loss:.4f}")

    elapsed = time.perf_counter() - t0
    metrics = {
        "config": cfg,
        "seed": seed,
        "device": device,
        "amp_dtype": str(amp_dtype) if use_amp else "fp32",
        "params": num_params,
        "tokens": {
            "train": int(len(train_tokens)),
            "valid": int(len(val_tokens)),
            "seen": max_steps * tokens_per_step,
        },
        "loss": {"train": losses, "val_steps": val_steps, "val": val_losses},
        "final": {"train": losses[-1], "val": val_losses[-1] if val_losses else None},
        "time_s": elapsed,
        "tokens_per_s": max_steps * tokens_per_step / elapsed,
    }
    return metrics, model
