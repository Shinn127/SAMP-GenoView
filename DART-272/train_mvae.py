from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dart272.data import HumanML3D272Dataset, collate_batch
from dart272.utils import ensure_dir, resolve_device, save_json, set_seed
from dart272.vae import AutoMldVae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="humanml3d_272")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--future-length", type=int, default=8)
    parser.add_argument("--num-primitives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--latent-size", type=int, default=1)
    parser.add_argument("--latent-width", type=int, default=128)
    parser.add_argument("--h-dim", type=int, default=512)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rollout-prob", type=float, default=0.5)
    return parser.parse_args()


def primitive_slice(motion: torch.Tensor, primitive_idx: int, history_length: int, future_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    start = primitive_idx * future_length
    history = motion[:, start : start + history_length].contiguous()
    future = motion[:, start + history_length : start + history_length + future_length].contiguous()
    return history, future


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = ensure_dir(args.save_dir)
    print(f"Using device: {device}")

    train_set = HumanML3D272Dataset(
        data_root=args.data_root,
        split="train",
        history_length=args.history_length,
        future_length=args.future_length,
        num_primitives=args.num_primitives,
    )
    val_set = HumanML3D272Dataset(
        data_root=args.data_root,
        split="val",
        history_length=args.history_length,
        future_length=args.future_length,
        num_primitives=args.num_primitives,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

    model = AutoMldVae(
        nfeats=train_set.feature_dim,
        latent_dim=(args.latent_size, args.latent_width),
        h_dim=args.h_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_json(
        save_dir / "config.json",
        {
            "data_root": args.data_root,
            "history_length": args.history_length,
            "future_length": args.future_length,
            "num_primitives": args.num_primitives,
            "feature_dim": train_set.feature_dim,
            "latent_size": args.latent_size,
            "latent_width": args.latent_width,
            "h_dim": args.h_dim,
            "ff_size": args.ff_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "device": str(device),
        },
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress, start=1):
            motion = batch["motion"].to(device)
            loss_total = 0.0
            last_primitive = None
            for primitive_idx in range(args.num_primitives):
                history_gt, future_gt = primitive_slice(motion, primitive_idx, args.history_length, args.future_length)
                history = last_primitive[:, -args.history_length :] if last_primitive is not None else history_gt
                history = history.contiguous()
                latent, dist = model.encode(future_gt, history)
                future_pred = model.decode(latent, history, args.future_length).contiguous()
                rec_loss = F.smooth_l1_loss(future_pred, future_gt.contiguous())
                kl_loss = torch.distributions.kl_divergence(
                    dist,
                    torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale)),
                ).mean()
                loss = rec_loss + args.kl_weight * kl_loss
                loss_total = loss_total + loss
                if primitive_idx < args.num_primitives - 1 and torch.rand(1).item() < args.rollout_prob:
                    last_primitive = future_pred.detach()
                else:
                    last_primitive = None
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss_total.item())
            progress.set_postfix(loss=f"{loss_total.item():.4f}")
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"val epoch {epoch}", leave=False), start=1):
                motion = batch["motion"].to(device)
                loss_total = 0.0
                for primitive_idx in range(args.num_primitives):
                    history, future = primitive_slice(motion, primitive_idx, args.history_length, args.future_length)
                    history = history.contiguous()
                    latent, dist = model.encode(future, history)
                    future_pred = model.decode(latent, history, args.future_length).contiguous()
                    rec_loss = F.smooth_l1_loss(future_pred, future.contiguous())
                    kl_loss = torch.distributions.kl_divergence(
                        dist,
                        torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale)),
                    ).mean()
                    loss_total = loss_total + rec_loss + args.kl_weight * kl_loss
                val_losses.append(loss_total.item())
                if args.max_val_batches and batch_idx >= args.max_val_batches:
                    break

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(payload, save_dir / "checkpoint_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, save_dir / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
