from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dart272.data import HumanML3D272Dataset, collate_batch
from dart272.denoiser import DenoiserMLP, DenoiserTransformer
from dart272.diffusion import GaussianDiffusion
from dart272.text import encode_text, load_and_freeze_clip
from dart272.utils import ensure_dir, load_json, resolve_device, save_json, set_seed
from dart272.vae import AutoMldVae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="humanml3d_272")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--mvae-ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    parser.add_argument("--denoiser-type", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--denoiser-h-dim", type=int, default=512)
    parser.add_argument("--denoiser-layers", type=int, default=6)
    parser.add_argument("--denoiser-heads", type=int, default=4)
    parser.add_argument("--denoiser-ff-size", type=int, default=1024)
    parser.add_argument("--denoiser-blocks", type=int, default=2)
    parser.add_argument("--cond-mask-prob", type=float, default=0.1)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--rollout-prob", type=float, default=0.5)
    return parser.parse_args()


def primitive_slice(motion: torch.Tensor, primitive_idx: int, history_length: int, future_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    start = primitive_idx * future_length
    history = motion[:, start : start + history_length].contiguous()
    future = motion[:, start + history_length : start + history_length + future_length].contiguous()
    return history, future


def build_denoiser(args: argparse.Namespace, history_shape: tuple[int, int], noise_shape: tuple[int, int]) -> torch.nn.Module:
    if args.denoiser_type == "mlp":
        return DenoiserMLP(
            h_dim=args.denoiser_h_dim,
            n_blocks=args.denoiser_blocks,
            clip_dim=args.text_dim,
            history_shape=history_shape,
            noise_shape=noise_shape,
            cond_mask_prob=args.cond_mask_prob,
        )
    return DenoiserTransformer(
        h_dim=args.denoiser_h_dim,
        ff_size=args.denoiser_ff_size,
        num_layers=args.denoiser_layers,
        num_heads=args.denoiser_heads,
        clip_dim=args.text_dim,
        history_shape=history_shape,
        noise_shape=noise_shape,
        cond_mask_prob=args.cond_mask_prob,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = ensure_dir(args.save_dir)
    print(f"Using device: {device}")

    mvae_ckpt = torch.load(args.mvae_ckpt, map_location="cpu")
    mvae_cfg = load_json(Path(args.mvae_ckpt).parent / "config.json")

    train_set = HumanML3D272Dataset(
        data_root=args.data_root,
        split="train",
        history_length=mvae_cfg["history_length"],
        future_length=mvae_cfg["future_length"],
        num_primitives=mvae_cfg["num_primitives"],
    )
    val_set = HumanML3D272Dataset(
        data_root=args.data_root,
        split="val",
        history_length=mvae_cfg["history_length"],
        future_length=mvae_cfg["future_length"],
        num_primitives=mvae_cfg["num_primitives"],
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

    vae = AutoMldVae(
        nfeats=mvae_cfg["feature_dim"],
        latent_dim=(mvae_cfg["latent_size"], mvae_cfg["latent_width"]),
        h_dim=mvae_cfg["h_dim"],
        ff_size=mvae_cfg["ff_size"],
        num_layers=mvae_cfg["num_layers"],
        num_heads=mvae_cfg["num_heads"],
        dropout=mvae_cfg["dropout"],
    ).to(device)
    vae.load_state_dict(mvae_ckpt["model_state"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    clip_model = load_and_freeze_clip(args.clip_version, device=device)
    denoiser = build_denoiser(
        args,
        history_shape=(mvae_cfg["history_length"], mvae_cfg["feature_dim"]),
        noise_shape=(mvae_cfg["latent_size"], mvae_cfg["latent_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(list(denoiser.parameters()), lr=args.lr)
    diffusion = GaussianDiffusion(num_steps=args.diffusion_steps)

    save_json(
        save_dir / "config.json",
        {
            "data_root": args.data_root,
            "history_length": mvae_cfg["history_length"],
            "future_length": mvae_cfg["future_length"],
            "num_primitives": mvae_cfg["num_primitives"],
            "feature_dim": mvae_cfg["feature_dim"],
            "latent_size": mvae_cfg["latent_size"],
            "latent_width": mvae_cfg["latent_width"],
            "text_dim": args.text_dim,
            "clip_version": args.clip_version,
            "denoiser_type": args.denoiser_type,
            "denoiser_h_dim": args.denoiser_h_dim,
            "denoiser_layers": args.denoiser_layers,
            "denoiser_heads": args.denoiser_heads,
            "denoiser_ff_size": args.denoiser_ff_size,
            "denoiser_blocks": args.denoiser_blocks,
            "diffusion_steps": args.diffusion_steps,
            "device": str(device),
        },
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}", leave=False), start=1):
            motion = batch["motion"].to(device)
            loss_total = 0.0
            last_primitive = None
            for primitive_idx in range(mvae_cfg["num_primitives"]):
                history_gt, future_gt = primitive_slice(motion, primitive_idx, mvae_cfg["history_length"], mvae_cfg["future_length"])
                history = last_primitive[:, -mvae_cfg["history_length"] :] if last_primitive is not None else history_gt
                history = history.contiguous()
                latent_gt, _ = vae.encode(future_gt, history)
                latent_gt = latent_gt.contiguous()
                x_start = latent_gt.permute(1, 0, 2).contiguous()
                t = torch.randint(0, diffusion.num_steps, (motion.shape[0],), device=device)
                x_t = diffusion.q_sample(x_start, t)
                texts = [sample_texts[primitive_idx] for sample_texts in batch["texts"]]
                text_embedding = encode_text(clip_model, texts, force_empty_zero=True)
                y = {
                    "text_embedding": text_embedding,
                    "history_motion_normalized": history,
                }
                x_start_pred = denoiser(x_t, t, y).contiguous()
                latent_pred = x_start_pred.permute(1, 0, 2).contiguous()
                future_pred = vae.decode(latent_pred, history, mvae_cfg["future_length"]).contiguous()
                latent_loss = F.smooth_l1_loss(latent_pred, latent_gt.contiguous())
                feature_loss = F.smooth_l1_loss(future_pred, future_gt.contiguous())
                loss = args.latent_weight * latent_loss + args.feature_weight * feature_loss
                loss_total = loss_total + loss
                if primitive_idx < mvae_cfg["num_primitives"] - 1 and torch.rand(1).item() < args.rollout_prob:
                    last_primitive = future_pred.detach()
                else:
                    last_primitive = None
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()), 1.0)
            optimizer.step()
            train_losses.append(loss_total.item())
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

        denoiser.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"val epoch {epoch}", leave=False), start=1):
                motion = batch["motion"].to(device)
                loss_total = 0.0
                for primitive_idx in range(mvae_cfg["num_primitives"]):
                    history, future = primitive_slice(motion, primitive_idx, mvae_cfg["history_length"], mvae_cfg["future_length"])
                    history = history.contiguous()
                    latent_gt, _ = vae.encode(future, history)
                    latent_gt = latent_gt.contiguous()
                    x_start = latent_gt.permute(1, 0, 2).contiguous()
                    t = torch.randint(0, diffusion.num_steps, (motion.shape[0],), device=device)
                    x_t = diffusion.q_sample(x_start, t)
                    texts = [sample_texts[primitive_idx] for sample_texts in batch["texts"]]
                    text_embedding = encode_text(clip_model, texts, force_empty_zero=True)
                    x_start_pred = denoiser(
                        x_t,
                        t,
                        {
                            "text_embedding": text_embedding,
                            "history_motion_normalized": history,
                        },
                    ).contiguous()
                    latent_pred = x_start_pred.permute(1, 0, 2).contiguous()
                    future_pred = vae.decode(latent_pred, history, mvae_cfg["future_length"]).contiguous()
                    latent_loss = F.smooth_l1_loss(latent_pred, latent_gt.contiguous())
                    feature_loss = F.smooth_l1_loss(future_pred, future.contiguous())
                    loss_total = loss_total + args.latent_weight * latent_loss + args.feature_weight * feature_loss
                val_losses.append(loss_total.item())
                if args.max_val_batches and batch_idx >= args.max_val_batches:
                    break

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")
        payload = {
            "epoch": epoch,
            "vae_checkpoint": str(args.mvae_ckpt),
            "denoiser_state": denoiser.state_dict(),
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
