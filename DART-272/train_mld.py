from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    # Denoiser architecture — default to transformer matching DART-main
    parser.add_argument("--denoiser-type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--denoiser-h-dim", type=int, default=512)
    parser.add_argument("--denoiser-layers", type=int, default=8)
    parser.add_argument("--denoiser-heads", type=int, default=4)
    parser.add_argument("--denoiser-ff-size", type=int, default=1024)
    parser.add_argument("--denoiser-blocks", type=int, default=2,
                        help="Only used for MLP denoiser")
    parser.add_argument("--cond-mask-prob", type=float, default=0.1)
    # Loss weights
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=1e4,
                        help="Weight for joints temporal delta consistency loss")
    parser.add_argument("--transl-delta-weight", type=float, default=1e4,
                        help="Weight for root translation delta consistency loss")
    parser.add_argument("--orient-delta-weight", type=float, default=1e4,
                        help="Weight for heading orientation delta consistency loss")
    # Three-stage curriculum learning
    parser.add_argument("--stage1-steps", type=int, default=100000,
                        help="Stage 1: pure single-step denoising, no rollout")
    parser.add_argument("--stage2-steps", type=int, default=100000,
                        help="Stage 2: linearly increasing rollout probability 0->1")
    parser.add_argument("--stage3-steps", type=int, default=100000,
                        help="Stage 3: full rollout only")
    parser.add_argument("--full-rollout", type=int, default=1,
                        help="If 1, use full DDPM sampling loop for rollout history in stage2/3")
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--val-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--anneal-lr", type=int, default=1,
                        help="Whether to linearly anneal learning rate to 0")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate. 0 to disable EMA.")
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
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


def get_rollout_prob(step: int, stage1_steps: int, stage2_steps: int) -> float:
    """Three-stage curriculum: stage1=no rollout, stage2=linear ramp, stage3=full rollout."""
    if step <= stage1_steps:
        return 0.0
    elif step <= stage1_steps + stage2_steps:
        progress = (step - stage1_steps) / max(float(stage2_steps), 1e-6)
        return min(1.0, progress)
    else:
        return 1.0


def temporal_delta_loss(pred: torch.Tensor, history_last_frame: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute temporal delta consistency losses on the 272-dim representation.

    272-dim layout (from representation_272.py):
      [0:2]     root XZ velocity (no heading)
      [2:8]     heading angular velocity (6D rotation)
      [8:74]    joint positions (22 joints * 3)
      [74:140]  joint velocities (22 joints * 3)
      [140:272] joint rotations 6D (22 joints * 6)

    Returns dict with three loss components.
    """
    seq = torch.cat([history_last_frame.unsqueeze(1), pred], dim=1).contiguous()

    # 1. Joint positions delta vs velocity
    pos = seq[:, :, 8:74].contiguous()
    calc_joints_delta = (pos[:, 1:, :] - pos[:, :-1, :]).contiguous()
    pred_joints_vel = pred[:, :, 74:140].contiguous()
    joints_delta_loss = F.smooth_l1_loss(calc_joints_delta, pred_joints_vel)

    # 2. Root translation delta
    root_pos = seq[:, :, 8:11].contiguous()
    calc_root_delta = (root_pos[:, 1:, :] - root_pos[:, :-1, :]).contiguous()
    pred_root_vel = pred[:, :, 74:77].contiguous()
    transl_delta_loss = F.smooth_l1_loss(calc_root_delta, pred_root_vel)

    # 3. Heading orientation smoothness
    heading_6d = pred[:, :, 2:8].contiguous()
    heading_jerk = (heading_6d[:, 2:, :] - 2 * heading_6d[:, 1:-1, :] + heading_6d[:, :-2, :]).contiguous()
    orient_delta_loss = F.smooth_l1_loss(heading_jerk, torch.zeros_like(heading_jerk))

    return {
        "joints_delta": joints_delta_loss,
        "transl_delta": transl_delta_loss,
        "orient_delta": orient_delta_loss,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = ensure_dir(args.save_dir)
    print(f"Using device: {device}")

    mvae_ckpt_path = Path(args.mvae_ckpt).expanduser().resolve()
    mvae_ckpt = torch.load(mvae_ckpt_path, map_location="cpu")
    mvae_cfg = load_json(mvae_ckpt_path.parent / "config.json")

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
    print(f"  VAE latent_mean={vae.latent_mean.item():.4f} latent_std={vae.latent_std.item():.4f}")

    clip_model = load_and_freeze_clip(args.clip_version, device=device)
    denoiser = build_denoiser(
        args,
        history_shape=(mvae_cfg["history_length"], mvae_cfg["feature_dim"]),
        noise_shape=(mvae_cfg["latent_size"], mvae_cfg["latent_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(list(denoiser.parameters()), lr=args.lr)
    diffusion = GaussianDiffusion(num_steps=args.diffusion_steps)

    # EMA model
    ema_denoiser = None
    if args.ema_decay > 0:
        ema_denoiser = copy.deepcopy(denoiser)
        ema_denoiser.eval()
        print(f"  EMA enabled with decay={args.ema_decay}")

    total_steps = args.stage1_steps + args.stage2_steps + args.stage3_steps

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt_resume = torch.load(args.resume, map_location="cpu")
        denoiser.load_state_dict(ckpt_resume["denoiser_state"])
        if "optimizer_state" in ckpt_resume:
            optimizer.load_state_dict(ckpt_resume["optimizer_state"])
        start_step = ckpt_resume.get("global_step", 0)
        if ema_denoiser is not None:
            ema_denoiser.load_state_dict(ckpt_resume["denoiser_state"])
        print(f"  Resumed from {args.resume} at step {start_step}")

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
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "stage3_steps": args.stage3_steps,
            "total_steps": total_steps,
            "full_rollout": args.full_rollout,
            "delta_weight": args.delta_weight,
            "device": str(device),
        },
    )

    best_val = float("inf")
    global_step = start_step
    epoch = 0
    writer = SummaryWriter(log_dir=save_dir / "tb_logs")
    progress_bar = tqdm(total=total_steps, initial=start_step, desc="training")

    while global_step < total_steps:
        epoch += 1
        denoiser.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            if global_step >= total_steps:
                break

            # Learning rate annealing
            if args.anneal_lr:
                frac = 1.0 - global_step / total_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * args.lr

            motion = batch["motion"].to(device)
            loss_total = 0.0
            last_primitive = None
            rollout_prob = get_rollout_prob(global_step, args.stage1_steps, args.stage2_steps)

            for primitive_idx in range(mvae_cfg["num_primitives"]):
                history_gt, future_gt = primitive_slice(motion, primitive_idx, mvae_cfg["history_length"], mvae_cfg["future_length"])
                history = last_primitive[:, -mvae_cfg["history_length"] :] if last_primitive is not None else history_gt
                history = history.contiguous()

                latent_gt, _ = vae.encode(future_gt, history, scale_latent=True)
                latent_gt = latent_gt.contiguous()
                x_start = latent_gt.permute(1, 0, 2).contiguous()

                t = torch.randint(0, diffusion.num_steps, (motion.shape[0],), device=device)
                x_t = diffusion.q_sample(x_start, t)

                texts = [sample_texts[primitive_idx] for sample_texts in batch["texts"]]
                if "text_embeddings" in batch:
                    text_embedding = batch["text_embeddings"][:, primitive_idx].to(device)
                else:
                    text_embedding = encode_text(clip_model, texts, force_empty_zero=True)
                y = {
                    "text_embedding": text_embedding,
                    "history_motion_normalized": history,
                }
                x_start_pred = denoiser(x_t, t, y).contiguous()
                latent_pred = x_start_pred.permute(1, 0, 2).contiguous()
                future_pred = vae.decode(latent_pred, history, mvae_cfg["future_length"], scale_latent=True).contiguous()

                latent_loss = F.smooth_l1_loss(latent_pred, latent_gt.contiguous())
                feature_loss = F.smooth_l1_loss(future_pred, future_gt.contiguous())

                # Temporal delta consistency loss on decoded features
                delta_total = torch.tensor(0.0, device=device)
                if args.delta_weight > 0 or args.transl_delta_weight > 0 or args.orient_delta_weight > 0:
                    delta_losses = temporal_delta_loss(
                        train_set.denormalize(future_pred),
                        train_set.denormalize(history[:, -1:, :]).squeeze(1),
                    )
                    delta_total = (args.delta_weight * delta_losses["joints_delta"]
                                   + args.transl_delta_weight * delta_losses["transl_delta"]
                                   + args.orient_delta_weight * delta_losses["orient_delta"])

                loss = args.latent_weight * latent_loss + args.feature_weight * feature_loss + delta_total
                loss_total = loss_total + loss

                # Decide whether to use rollout for next primitive
                if primitive_idx < mvae_cfg["num_primitives"] - 1 and torch.rand(1).item() < rollout_prob:
                    # Full rollout: use complete DDPM sampling loop (matches inference)
                    if args.full_rollout:
                        with torch.no_grad():
                            y_rollout = {
                                "text_embedding": text_embedding,
                                "history_motion_normalized": history,
                            }
                            x_sampled = diffusion.p_sample_loop(
                                denoiser,
                                shape=x_start.shape,
                                model_kwargs={"y": y_rollout},
                                device=device,
                            )
                            latent_sampled = x_sampled.permute(1, 0, 2)
                            future_sampled = vae.decode(latent_sampled, history, mvae_cfg["future_length"], scale_latent=True)
                        last_primitive = future_sampled.detach()
                    else:
                        last_primitive = future_pred.detach()
                else:
                    last_primitive = None

                global_step += 1

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()), 1.0)
            optimizer.step()

            # Update EMA
            if ema_denoiser is not None:
                with torch.no_grad():
                    for param, ema_param in zip(denoiser.parameters(), ema_denoiser.parameters()):
                        ema_param.data.mul_(args.ema_decay).add_(param.data, alpha=1.0 - args.ema_decay)

            progress_bar.update(mvae_cfg["num_primitives"])
            progress_bar.set_postfix(
                loss=f"{loss_total.item():.4f}",
                rollout_p=f"{rollout_prob:.2f}",
                stage=1 if global_step <= args.stage1_steps else (2 if global_step <= args.stage1_steps + args.stage2_steps else 3),
            )

            # Logging
            if global_step % args.log_interval < mvae_cfg["num_primitives"]:
                stage = 1 if global_step <= args.stage1_steps else (2 if global_step <= args.stage1_steps + args.stage2_steps else 3)
                writer.add_scalar("train/loss", loss_total.item(), global_step)
                writer.add_scalar("train/rollout_prob", rollout_prob, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/stage", stage, global_step)
                print(f"  [step {global_step}/{total_steps}] stage={stage} rollout_prob={rollout_prob:.3f} loss={loss_total.item():.4f}")

            # Validation
            if global_step % args.val_interval < mvae_cfg["num_primitives"]:
                save_denoiser = ema_denoiser if ema_denoiser is not None else denoiser
                val_loss = _validate(save_denoiser, vae, diffusion, clip_model, val_loader, mvae_cfg, train_set, args, device)
                print(f"  [step {global_step}] val_loss={val_loss:.6f}")
                writer.add_scalar("val/loss", val_loss, global_step)
                denoiser.train()
                payload = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "vae_checkpoint": str(mvae_ckpt_path),
                    "denoiser_state": save_denoiser.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "ema": ema_denoiser is not None,
                }
                torch.save(payload, save_dir / "checkpoint_last.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(payload, save_dir / "checkpoint_best.pt")

            # Save checkpoint
            if global_step % args.save_interval < mvae_cfg["num_primitives"]:
                save_denoiser = ema_denoiser if ema_denoiser is not None else denoiser
                payload = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "vae_checkpoint": str(mvae_ckpt_path),
                    "denoiser_state": save_denoiser.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "ema": ema_denoiser is not None,
                }
                torch.save(payload, save_dir / f"checkpoint_{global_step}.pt")
                print(f"  saved checkpoint_{global_step}.pt")

            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

    progress_bar.close()

    # Final save
    save_denoiser = ema_denoiser if ema_denoiser is not None else denoiser
    val_loss = _validate(save_denoiser, vae, diffusion, clip_model, val_loader, mvae_cfg, train_set, args, device)
    print(f"[final] step={global_step} val_loss={val_loss:.6f}")
    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "vae_checkpoint": str(mvae_ckpt_path),
        "denoiser_state": save_denoiser.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "ema": ema_denoiser is not None,
    }
    torch.save(payload, save_dir / "checkpoint_last.pt")
    if val_loss < best_val:
        torch.save(payload, save_dir / "checkpoint_best.pt")

    writer.close()


def _validate(
    denoiser: torch.nn.Module,
    vae: AutoMldVae,
    diffusion: GaussianDiffusion,
    clip_model: torch.nn.Module,
    val_loader: DataLoader,
    mvae_cfg: dict,
    dataset: HumanML3D272Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    denoiser.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            motion = batch["motion"].to(device)
            loss_total = 0.0
            for primitive_idx in range(mvae_cfg["num_primitives"]):
                history, future = primitive_slice(motion, primitive_idx, mvae_cfg["history_length"], mvae_cfg["future_length"])
                history = history.contiguous()
                latent_gt, _ = vae.encode(future, history, scale_latent=True)
                latent_gt = latent_gt.contiguous()
                x_start = latent_gt.permute(1, 0, 2).contiguous()
                t = torch.randint(0, diffusion.num_steps, (motion.shape[0],), device=device)
                x_t = diffusion.q_sample(x_start, t)
                texts = [sample_texts[primitive_idx] for sample_texts in batch["texts"]]
                if "text_embeddings" in batch:
                    text_embedding = batch["text_embeddings"][:, primitive_idx].to(device)
                else:
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
                future_pred = vae.decode(latent_pred, history, mvae_cfg["future_length"], scale_latent=True).contiguous()
                latent_loss = F.smooth_l1_loss(latent_pred, latent_gt.contiguous())
                feature_loss = F.smooth_l1_loss(future_pred, future.contiguous())
                delta_total = torch.tensor(0.0, device=device)
                if args.delta_weight > 0 or args.transl_delta_weight > 0 or args.orient_delta_weight > 0:
                    delta_losses = temporal_delta_loss(
                        dataset.denormalize(future_pred),
                        dataset.denormalize(history[:, -1:, :]).squeeze(1),
                    )
                    delta_total = (args.delta_weight * delta_losses["joints_delta"]
                                   + args.transl_delta_weight * delta_losses["transl_delta"]
                                   + args.orient_delta_weight * delta_losses["orient_delta"])
                loss_total = loss_total + args.latent_weight * latent_loss + args.feature_weight * feature_loss + delta_total
            val_losses.append(loss_total.item())
            if args.max_val_batches and batch_idx >= args.max_val_batches:
                break
    return sum(val_losses) / max(len(val_losses), 1)


if __name__ == "__main__":
    main()
