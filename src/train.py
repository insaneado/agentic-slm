"""
Pre-training loop for the SLM.

Usage:
    python -m src.data           # build the token shards first
    python -m src.train          # then train

Checkpoints the best validation loss to `best_model_params.pt`.
"""

import argparse
import os
from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from .data import get_batch
from .model import GPT, GPTConfig


def build_argparser():
    ap = argparse.ArgumentParser(description="Train the TinyStories SLM.")
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    # NOTE: min_lr must stay BELOW learning_rate, otherwise the cosine schedule
    # anneals *upwards* over training instead of decaying.
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--eval-iters", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--grad-accum-steps", type=int, default=32)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="best_model_params.pt")
    ap.add_argument("--seed", type=int, default=42)
    return ap


@torch.no_grad()
def estimate_loss(model, args, device, ctx):
    """Average loss over `eval_iters` batches for both splits."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split, args.block_size, args.batch_size, device, args.data_dir)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(argv=None):
    args = build_argparser().parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    torch.manual_seed(args.seed)

    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif device_type == "cuda":
        dtype = "float16"
    else:
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    config = GPTConfig(block_size=args.block_size)
    model = GPT(config).to(device)
    print(f"device={device} dtype={dtype} params={model.num_parameters()/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, total_iters=args.warmup_steps),
            CosineAnnealingLR(
                optimizer, T_max=args.max_iters - args.warmup_steps, eta_min=args.min_lr
            ),
        ],
        milestones=[args.warmup_steps],
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for step in tqdm(range(args.max_iters)):
        if step % args.eval_iters == 0 and step != 0:
            losses = estimate_loss(model, args, device, ctx)
            print(f"step {step}: train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), args.out)
                print(f"  -> new best, saved {args.out}")

        X, y = get_batch("train", args.block_size, args.batch_size, device, args.data_dir)
        with ctx:
            _, loss = model(X, y)
            loss = loss / args.grad_accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == args.max_iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    print(f"done. best val loss {best_val_loss:.4f} -> {args.out}")
    return train_losses, val_losses


if __name__ == "__main__":
    main()
