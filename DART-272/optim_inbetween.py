"""Motion In-betweening via latent optimization (aligned with DART-main).

Given a start pose (history), a goal keyframe, text prompt, and duration,
optimize the initial noise for each motion primitive so that the generated
sequence smoothly transitions from start to goal while following the text.

Key alignment with DART-main:
  - Single noise tensor (not per-primitive list)
  - DDIM sampling (deterministic, differentiable)
  - Classifier-free guidance during optimization
  - Unit gradient normalization
  - Learning rate annealing
  - HuberLoss for goal matching

Usage:
    python DART-272/optim_inbetween.py \
        --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
        --mvae-ckpt DART-272/outputs/mvae_run2/checkpoint_best.pt \
        --start-motion humanml3d_272/motion_data/000962.npy \
        --start-frame 0 \
        --goal-motion humanml3d_272/motion_data/000962.npy \
        --goal-frame 80 \
        --text-prompt "walk forward" \
        --num-primitives 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dart272.denoiser import ClassifierFreeGuidanceWrapper, DenoiserMLP, DenoiserTransformer
from dart272.diffusion import GaussianDiffusion, extract
from dart272.text import encode_text, load_and_freeze_clip
from dart272.utils import ensure_dir, load_json, resolve_device, set_seed
from dart272.vae import AutoMldVae
from dart272.world_transform import local_to_world_joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="MLD checkpoint path")
    parser.add_argument("--mvae-ckpt", type=str, default=None,
                        help="Override VAE checkpoint path")
    parser.add_argument("--start-motion", type=str, required=True,
                        help="Path to .npy motion file for start pose")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--goal-frames", type=str, nargs="+", required=True,
                        help="Goal specifications as 'motion_path:source_frame:gen_frame'. "
                             "motion_path = .npy file to extract target pose from. "
                             "source_frame = frame index in that file ('last' for last frame). "
                             "gen_frame = frame index in generated sequence where this pose should appear. "
                             "Example: --goal-frames humanml3d_272/motion_data/000962.npy:120:120 "
                             "humanml3d_272/motion_data/000003.npy:50:200")
    parser.add_argument("--text-prompt", type=str, default="walk forward")
    parser.add_argument("--num-primitives", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output filename stem (without extension). Default: auto-generated.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    # Optimization parameters (aligned with DART-main defaults)
    parser.add_argument("--optim-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--anneal-lr", type=int, default=1,
                        help="Linearly anneal learning rate to 0")
    parser.add_argument("--unit-grad", type=int, default=1,
                        help="Normalize gradient to unit norm (DART-main key technique)")
    parser.add_argument("--init-noise-scale", type=float, default=1.0,
                        help="Scale of initial noise")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--weight-jerk", type=float, default=0.0,
                        help="Weight for jerk (smoothness) loss")
    parser.add_argument("--weight-floor", type=float, default=0.0,
                        help="Weight for floor penetration loss")
    parser.add_argument("--ddim-steps", type=int, default=10,
                        help="Number of DDIM steps for sampling")
    return parser.parse_args()


def build_denoiser(cfg: dict, cond_mask_prob: float = 0.1) -> torch.nn.Module:
    history_shape = (cfg["history_length"], cfg["feature_dim"])
    noise_shape = (cfg["latent_size"], cfg["latent_width"])
    if cfg["denoiser_type"] == "mlp":
        return DenoiserMLP(
            h_dim=cfg["denoiser_h_dim"],
            n_blocks=cfg["denoiser_blocks"],
            clip_dim=cfg["text_dim"],
            history_shape=history_shape,
            noise_shape=noise_shape,
            cond_mask_prob=cond_mask_prob,
        )
    return DenoiserTransformer(
        h_dim=cfg["denoiser_h_dim"],
        ff_size=cfg["denoiser_ff_size"],
        num_layers=cfg["denoiser_layers"],
        num_heads=cfg["denoiser_heads"],
        clip_dim=cfg["text_dim"],
        history_shape=history_shape,
        noise_shape=noise_shape,
        cond_mask_prob=cond_mask_prob,
    )


def ddim_sample_full_chain(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    init_noise: torch.Tensor,
    model_kwargs: dict,
    device: torch.device,
    ddim_steps: int = 10,
) -> torch.Tensor:
    """DDIM sampling with fixed init noise (differentiable).

    Matches DART-main's ddim_sample_loop_full_chain: deterministic DDIM
    starting from the provided noise, returning the final x_start prediction.
    """
    num_steps = diffusion.num_steps
    # Create evenly spaced timestep subsequence
    step_indices = torch.linspace(0, num_steps - 1, ddim_steps, dtype=torch.long, device=device)

    x_t = init_noise
    for i in reversed(range(ddim_steps)):
        t = torch.full((x_t.shape[0],), step_indices[i].item(), device=device, dtype=torch.long)

        # Model predicts x_start
        x_start = model(x_t, t, model_kwargs["y"]).clamp(-5.0, 5.0)

        if i > 0:
            t_prev = torch.full((x_t.shape[0],), step_indices[i - 1].item(), device=device, dtype=torch.long)
            alpha_t = extract(diffusion.alphas_cumprod, t, x_t.shape)
            alpha_prev = extract(diffusion.alphas_cumprod, t_prev, x_t.shape)

            # DDIM deterministic step
            eps_pred = (x_t - torch.sqrt(alpha_t) * x_start) / torch.sqrt((1.0 - alpha_t).clamp(min=1e-8))
            x_t = torch.sqrt(alpha_prev) * x_start + torch.sqrt((1.0 - alpha_prev).clamp(min=0)) * eps_pred
        else:
            x_t = x_start

    return x_t


def calc_jerk(positions: torch.Tensor) -> torch.Tensor:
    """Compute max jerk across joints and frames (DART-main style).

    Args:
        positions: [B, T, num_joints * 3] or [B, T, num_joints, 3]
    """
    if positions.ndim == 3:
        # Reshape to [B, T, J, 3]
        positions = positions.reshape(positions.shape[0], positions.shape[1], -1, 3)
    vel = positions[:, 1:] - positions[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk_norm = torch.sqrt((jerk ** 2).sum(dim=-1))  # [B, T-3, J]
    return jerk_norm.amax(dim=[1, 2]).mean()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Load models
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_json(checkpoint_path.parent / "config.json")

    # Resolve VAE checkpoint
    if args.mvae_ckpt:
        mvae_ckpt_path = Path(args.mvae_ckpt)
    else:
        mvae_ckpt_path = Path(ckpt["vae_checkpoint"])
        if not mvae_ckpt_path.exists():
            candidate = checkpoint_path.parent.parent / mvae_ckpt_path.parent.name / mvae_ckpt_path.name
            if candidate.exists():
                mvae_ckpt_path = candidate
            else:
                raise FileNotFoundError(
                    f"Cannot find VAE checkpoint. Use --mvae-ckpt to specify."
                )
    mvae_cfg = load_json(mvae_ckpt_path.parent / "config.json")
    mvae_ckpt = torch.load(mvae_ckpt_path, map_location="cpu")
    print(f"  VAE loaded from: {mvae_ckpt_path}")

    history_length = cfg["history_length"]
    future_length = cfg["future_length"]
    feature_dim = cfg["feature_dim"]
    num_primitives = args.num_primitives

    # Normalize/denormalize
    data_root = Path(cfg["data_root"])
    mean = torch.from_numpy(np.load(data_root / "mean_std" / "Mean.npy")).float().to(device)
    std = torch.from_numpy(np.load(data_root / "mean_std" / "Std.npy")).float().clamp(min=1e-6).to(device)

    def normalize(m: torch.Tensor) -> torch.Tensor:
        return (m - mean) / std

    def denormalize(m: torch.Tensor) -> torch.Tensor:
        return m * std + mean

    # VAE
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
    for p in vae.parameters():
        p.requires_grad = False

    # Denoiser with CFG wrapper (aligned with DART-main)
    denoiser = build_denoiser(cfg, cond_mask_prob=0.1).to(device)
    denoiser.load_state_dict(ckpt["denoiser_state"])
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False
    model = ClassifierFreeGuidanceWrapper(denoiser, guidance_scale=args.guidance_scale)

    # CLIP
    clip_model = load_and_freeze_clip(cfg.get("clip_version", "ViT-B/32"), device=device)

    # Diffusion
    diffusion = GaussianDiffusion(num_steps=cfg["diffusion_steps"])

    # Load start motion
    start_raw = torch.from_numpy(np.load(args.start_motion)).float().to(device)
    history = normalize(start_raw[args.start_frame : args.start_frame + history_length]).unsqueeze(0)

    # Parse goal specifications: "motion_path:source_frame:gen_frame"
    total_gen_frames = num_primitives * future_length
    goal_targets = []  # list of (gen_frame_idx, target_joints [1, 22, 3])
    goal_motion_cache = {}  # cache loaded motions to avoid re-loading

    for spec in args.goal_frames:
        parts = spec.rsplit(":", 2)  # split from right to handle paths with colons
        if len(parts) != 3:
            raise ValueError(f"Invalid goal spec '{spec}'. Expected 'motion_path:source_frame:gen_frame'")
        motion_path, src_str, gen_str = parts

        # Load and cache motion
        if motion_path not in goal_motion_cache:
            raw = torch.from_numpy(np.load(motion_path)).float().to(device)
            world_joints = local_to_world_joints(raw.unsqueeze(0))  # [1, T, 22, 3]
            goal_motion_cache[motion_path] = (raw, world_joints)
        raw, world_joints = goal_motion_cache[motion_path]

        # Resolve source frame
        if src_str.strip() == "last":
            src_frame = raw.shape[0] - 1
        else:
            src_frame = int(src_str)
            if src_frame < 0:
                src_frame = raw.shape[0] + src_frame
        src_frame = min(src_frame, raw.shape[0] - 1)

        # Resolve gen frame
        if gen_str.strip() == "last":
            gen_frame = total_gen_frames - 1
        else:
            gen_frame = int(gen_str)
            if gen_frame < 0:
                gen_frame = total_gen_frames + gen_frame
        gen_frame = min(gen_frame, total_gen_frames - 1)

        target_joints = world_joints[:, src_frame]  # [1, 22, 3]
        goal_targets.append((gen_frame, target_joints))
        pelvis_pos = target_joints[0, 0].tolist()
        print(f"  Goal: {motion_path} frame {src_frame} → gen frame {gen_frame}, "
              f"pelvis at [{pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f}]")

    # Text embedding
    text_embedding = encode_text(clip_model, [args.text_prompt], force_empty_zero=True)

    # Initialize noise: single tensor [num_primitives, B, latent_size, latent_width]
    noise_shape = (cfg["latent_size"], cfg["latent_width"])
    noise = torch.randn(num_primitives, 1, *noise_shape, device=device) * args.init_noise_scale
    noise.requires_grad_(True)

    optimizer = torch.optim.Adam([noise], lr=args.lr)
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    print(f"  Optimizing {num_primitives} primitives × {args.optim_steps} steps")
    print(f"  Goals: {len(goal_targets)} keyframes")
    print(f"  Text: '{args.text_prompt}'")
    print(f"  DDIM steps: {args.ddim_steps}, guidance: {args.guidance_scale}")
    print(f"  Unit grad: {bool(args.unit_grad)}, anneal LR: {bool(args.anneal_lr)}")

    # Rollout function
    def rollout(noise_tensor: torch.Tensor) -> torch.Tensor:
        """Generate full sequence from noise tensor. Returns denormalized [1, T, 272]."""
        current_history = history
        all_futures = []

        for prim_idx in range(num_primitives):
            y = {
                "text_embedding": text_embedding,
                "history_motion_normalized": current_history,
                "scale": torch.ones(1, *noise_shape, device=device) * args.guidance_scale,
            }
            latent = ddim_sample_full_chain(
                model, diffusion, noise_tensor[prim_idx],
                model_kwargs={"y": y}, device=device,
                ddim_steps=args.ddim_steps,
            )
            latent_seq = latent.permute(1, 0, 2)
            future = vae.decode(latent_seq, current_history, future_length, scale_latent=True)
            all_futures.append(future)
            all_frames = torch.cat([current_history, future], dim=1)
            current_history = all_frames[:, -history_length:]

        generated = torch.cat(all_futures, dim=1)  # [1, num_prim * future_length, 272]
        return denormalize(generated)

    # Optimization loop
    for step in tqdm(range(args.optim_steps), desc="optimizing"):
        optimizer.zero_grad()

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - step / args.optim_steps
            optimizer.param_groups[0]["lr"] = frac * args.lr

        generated_denorm = rollout(noise)

        # --- Losses (aligned with DART-main) ---

        # 1. Goal keyframe matching (world-space joint positions, HuberLoss)
        world_joints = local_to_world_joints(generated_denorm)  # [1, T, 22, 3]
        loss_goal = torch.tensor(0.0, device=device)
        for gen_frame, target_joints in goal_targets:
            pred_joints = world_joints[:, gen_frame]  # [1, 22, 3]
            loss_goal = loss_goal + criterion(pred_joints, target_joints)
        loss_goal = loss_goal / len(goal_targets)

        # 2. Jerk loss on joint positions [8:74]
        loss_jerk = torch.tensor(0.0, device=device)
        if args.weight_jerk > 0:
            positions = generated_denorm[:, :, 8:74]  # [1, T, 66]
            loss_jerk = calc_jerk(positions)

        # 3. Floor loss (foot joints should not go below floor)
        loss_floor = torch.tensor(0.0, device=device)
        if args.weight_floor > 0:
            # Foot joints: indices 10, 11 (L_Foot, R_Foot) → dims [8+10*3:8+12*3] = [38:44]
            foot_positions = generated_denorm[:, :, 38:44].reshape(1, -1, 2, 3)  # [1, T, 2, 3]
            foot_height = foot_positions[:, :, :, 1]  # Y axis (height in root-relative)
            # First frame floor height as reference
            floor_ref = foot_height[:, 0].amin(dim=-1, keepdim=True)  # [1, 1]
            loss_floor = -(foot_height.amin(dim=-1) - floor_ref).clamp(max=0).mean()

        total_loss = loss_goal + args.weight_jerk * loss_jerk + args.weight_floor * loss_floor
        total_loss.backward()

        # Unit gradient normalization (DART-main key technique)
        if args.unit_grad and noise.grad is not None:
            grad_norm = noise.grad.norm(p=2, dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
            noise.grad.data /= grad_norm

        optimizer.step()

        if step % 50 == 0 or step == args.optim_steps - 1:
            print(f"  [step {step}] total={total_loss.item():.4f} "
                  f"goal={loss_goal.item():.6f} jerk={loss_jerk.item():.6f} "
                  f"floor={loss_floor.item():.6f}")

    # Final generation
    with torch.no_grad():
        generated_denorm = rollout(noise)
        history_denorm = denormalize(history)
        full_sequence = torch.cat([history_denorm, generated_denorm], dim=1)
        full_sequence_np = full_sequence.squeeze(0).cpu().numpy()

    # Save
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / "inbetween")
    stem = args.output_name or f"inbetween_{args.text_prompt.replace(' ', '_')[:30]}_p{num_primitives}"
    np.save(output_dir / f"{stem}.npy", full_sequence_np)

    metadata = {
        "text_prompt": args.text_prompt,
        "num_primitives": num_primitives,
        "optim_steps": args.optim_steps,
        "start_motion": args.start_motion,
        "start_frame": args.start_frame,
        "goal_specs": args.goal_frames,
        "goal_gen_frames": [gf for gf, _ in goal_targets],
        "shape": list(full_sequence_np.shape),
        "final_goal_loss": float(loss_goal.item()),
        "guidance_scale": args.guidance_scale,
        "ddim_steps": args.ddim_steps,
        "unit_grad": bool(args.unit_grad),
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output_dir / f'{stem}.npy'}")
    print(f"  Shape: {full_sequence_np.shape}")
    print(f"  Final goal loss: {loss_goal.item():.6f}")


if __name__ == "__main__":
    main()
