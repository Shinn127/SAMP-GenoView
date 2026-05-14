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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--delta-weight", type=float, default=100.0,
                        help="Weight for joints temporal delta consistency loss")
    parser.add_argument("--transl-delta-weight", type=float, default=100.0,
                        help="Weight for root translation delta consistency loss")
    parser.add_argument("--orient-delta-weight", type=float, default=100.0,
                        help="Weight for heading orientation delta consistency loss")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--latent-size", type=int, default=1)
    parser.add_argument("--latent-width", type=int, default=256)
    parser.add_argument("--h-dim", type=int, default=256)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Three-stage curriculum learning
    parser.add_argument("--stage1-steps", type=int, default=100000,
                        help="Stage 1: pure reconstruction, no rollout")
    parser.add_argument("--stage2-steps", type=int, default=50000,
                        help="Stage 2: linearly increasing rollout probability 0->1")
    parser.add_argument("--stage3-steps", type=int, default=50000,
                        help="Stage 3: full rollout only")
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--val-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--anneal-lr", type=int, default=1,
                        help="Whether to linearly anneal learning rate to 0")
    parser.add_argument("--latent-scale-batches", type=int, default=10,
                        help="Number of batches to estimate latent mean/std (0=all)")
    # EMA
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
      [2:8]     heading angular velocity (6D rotation of frame-to-frame heading diff)
      [8:74]    joint positions (22 joints * 3, no heading, at XZ origin)
      [74:140]  joint velocities (22 joints * 3)
      [140:272] joint rotations 6D (22 joints * 6)

    Returns dict with three loss components:
      - joints_delta: position[t] - position[t-1] should match velocity[t]
      - transl_delta: root XZ position diff should match root XZ velocity
      - orient_delta: heading rotation consistency (6D)
    """
    # Concatenate last history frame with predicted future: [B, F+1, D]
    seq = torch.cat([history_last_frame.unsqueeze(1), pred], dim=1).contiguous()

    # 1. Joint positions delta vs velocity
    pos = seq[:, :, 8:74].contiguous()  # [B, F+1, 66]
    calc_joints_delta = (pos[:, 1:, :] - pos[:, :-1, :]).contiguous()  # [B, F, 66]
    pred_joints_vel = pred[:, :, 74:140].contiguous()  # [B, F, 66]
    joints_delta_loss = F.smooth_l1_loss(calc_joints_delta, pred_joints_vel)

    # 2. Root XZ translation delta vs root velocity
    # Root position is at joint index 0 within positions: [8:11] (x,y,z of root)
    # But positions are already at XZ origin per frame, so root XZ is always 0.
    # The root XZ velocity [0:2] represents the actual displacement.
    # We check: root_pos_xz[t] - root_pos_xz[t-1] should be ~0 (since positions are re-centered)
    # and root_vel[0:2] encodes the actual world displacement.
    # For consistency: velocity of root joint in local frame [74:77] should match position delta [8:11]
    root_pos = seq[:, :, 8:11].contiguous()  # [B, F+1, 3] (root joint position)
    calc_root_delta = (root_pos[:, 1:, :] - root_pos[:, :-1, :]).contiguous()  # [B, F, 3]
    pred_root_vel = pred[:, :, 74:77].contiguous()  # [B, F, 3] (root joint velocity)
    transl_delta_loss = F.smooth_l1_loss(calc_root_delta, pred_root_vel)

    # 3. Heading orientation delta consistency
    # [2:8] is the 6D rotation of heading change. For the first frame it's identity [1,0,0,0,1,0].
    # We can't easily compute a "ground truth" heading delta from positions alone,
    # but we can enforce smoothness: the predicted heading delta should be temporally smooth.
    heading_6d = pred[:, :, 2:8].contiguous()  # [B, F, 6]
    heading_jerk = (heading_6d[:, 2:, :] - 2 * heading_6d[:, 1:-1, :] + heading_6d[:, :-2, :]).contiguous()
    orient_delta_loss = F.smooth_l1_loss(heading_jerk, torch.zeros_like(heading_jerk))

    return {
        "joints_delta": joints_delta_loss,
        "transl_delta": transl_delta_loss,
        "orient_delta": orient_delta_loss,
    }


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    """Update EMA model parameters."""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


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

    # EMA model
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        print(f"  EMA enabled with decay={args.ema_decay}")

    total_steps = args.stage1_steps + args.stage2_steps + args.stage3_steps

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("global_step", 0)
        if ema_model is not None:
            ema_model.load_state_dict(ckpt["model_state"])
        print(f"  Resumed from {args.resume} at step {start_step}")

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
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "stage3_steps": args.stage3_steps,
            "total_steps": total_steps,
            "ema_decay": args.ema_decay,
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
        model.train()
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

                # Temporal delta consistency loss
                delta_losses = temporal_delta_loss(
                    train_set.denormalize(future_pred),
                    train_set.denormalize(history[:, -1:, :]).squeeze(1),
                )

                loss = (rec_loss + args.kl_weight * kl_loss
                        + args.delta_weight * delta_losses["joints_delta"]
                        + args.transl_delta_weight * delta_losses["transl_delta"]
                        + args.orient_delta_weight * delta_losses["orient_delta"])
                loss_total = loss_total + loss

                # Decide whether to use rollout for next primitive
                if primitive_idx < args.num_primitives - 1 and torch.rand(1).item() < rollout_prob:
                    last_primitive = future_pred.detach()
                else:
                    last_primitive = None

                global_step += 1

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA
            if ema_model is not None:
                update_ema(model, ema_model, args.ema_decay)

            progress_bar.update(args.num_primitives)
            progress_bar.set_postfix(
                loss=f"{loss_total.item():.4f}",
                rollout_p=f"{rollout_prob:.2f}",
                stage=1 if global_step <= args.stage1_steps else (2 if global_step <= args.stage1_steps + args.stage2_steps else 3),
            )

            # Logging
            if global_step % args.log_interval < args.num_primitives:
                stage = 1 if global_step <= args.stage1_steps else (2 if global_step <= args.stage1_steps + args.stage2_steps else 3)
                writer.add_scalar("train/loss", loss_total.item(), global_step)
                writer.add_scalar("train/rollout_prob", rollout_prob, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/stage", stage, global_step)
                print(f"  [step {global_step}/{total_steps}] stage={stage} rollout_prob={rollout_prob:.3f} loss={loss_total.item():.4f}")

            # Validation & save
            if global_step % args.val_interval < args.num_primitives:
                save_model = ema_model if ema_model is not None else model
                val_loss = _validate(save_model, val_loader, args, device, train_set)
                print(f"  [step {global_step}] val_loss={val_loss:.6f}")
                writer.add_scalar("val/loss", val_loss, global_step)
                l_mean, l_std = save_model.fit_latent_scale(
                    loader=val_loader,
                    device=device,
                    history_length=args.history_length,
                    future_length=args.future_length,
                    num_primitives=args.num_primitives,
                    max_batches=args.latent_scale_batches,
                )
                print(f"  [step {global_step}] latent_mean={l_mean:.4f} latent_std={l_std:.4f}")
                model.train()
                payload = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state": save_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "latent_mean": l_mean,
                    "latent_std": l_std,
                    "ema": ema_model is not None,
                }
                torch.save(payload, save_dir / "checkpoint_last.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(payload, save_dir / "checkpoint_best.pt")

            # Save numbered checkpoint
            if global_step % args.save_interval < args.num_primitives:
                save_model = ema_model if ema_model is not None else model
                payload = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state": save_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "latent_mean": float(save_model.latent_mean.item()),
                    "latent_std": float(save_model.latent_std.item()),
                    "ema": ema_model is not None,
                }
                torch.save(payload, save_dir / f"checkpoint_{global_step}.pt")
                print(f"  saved checkpoint_{global_step}.pt")

            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

    progress_bar.close()

    # Final save
    save_model = ema_model if ema_model is not None else model
    val_loss = _validate(save_model, val_loader, args, device, train_set)
    l_mean, l_std = save_model.fit_latent_scale(
        loader=val_loader,
        device=device,
        history_length=args.history_length,
        future_length=args.future_length,
        num_primitives=args.num_primitives,
        max_batches=args.latent_scale_batches,
    )
    print(f"[final] step={global_step} val_loss={val_loss:.6f} latent_mean={l_mean:.4f} latent_std={l_std:.4f}")
    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "model_state": save_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "latent_mean": l_mean,
        "latent_std": l_std,
        "ema": ema_model is not None,
    }
    torch.save(payload, save_dir / "checkpoint_last.pt")
    if val_loss < best_val:
        torch.save(payload, save_dir / "checkpoint_best.pt")

    writer.close()


def _validate(model: AutoMldVae, val_loader: DataLoader, args: argparse.Namespace,
              device: torch.device, dataset: HumanML3D272Dataset) -> float:
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
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
                delta_losses = temporal_delta_loss(
                    dataset.denormalize(future_pred),
                    dataset.denormalize(history[:, -1:, :]).squeeze(1),
                )
                loss_total = loss_total + (rec_loss + args.kl_weight * kl_loss
                             + args.delta_weight * delta_losses["joints_delta"]
                             + args.transl_delta_weight * delta_losses["transl_delta"]
                             + args.orient_delta_weight * delta_losses["orient_delta"])
            val_losses.append(loss_total.item())
            if args.max_val_batches and batch_idx >= args.max_val_batches:
                break
    return sum(val_losses) / max(len(val_losses), 1)


if __name__ == "__main__":
    main()