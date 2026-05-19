"""Vectorized RL environment wrapping the DART-272 motion pipeline.

This environment generates motion primitives (8 frames at 30fps) using the
pretrained VAE + Denoiser + Diffusion pipeline, and computes rewards based on
goal-reaching behavior using world-space joint positions extracted directly
from the 272-dim representation via world_transform.

Each RL step:
  1. Policy outputs a 256D latent noise action
  2. DDIM sampling with classifier-free guidance produces a denoised latent
  3. VAE decoder produces 8 frames of 272-dim motion
  4. World transform converts to world-space joints for reward computation
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from control.config import EnvArgs, GoalSchedulerArgs, RewardWeights
from control.goal_scheduler import GoalScheduler
from control.reward import compute_reward
from dart272.denoiser import (
    ClassifierFreeGuidanceWrapper,
    DenoiserMLP,
    DenoiserTransformer,
)
from dart272.diffusion import GaussianDiffusion
from dart272.text import encode_text, load_and_freeze_clip
from dart272.vae import AutoMldVae
from dart272.world_transform import (
    cumulative_rotation_batched,
    local_to_world_joints,
)


class Dart272Env:
    """Vectorized RL environment wrapping the DART-272 motion pipeline.

    Supports parallel rollouts via batch-dimension tensors. Each environment
    step generates one motion primitive (8 frames) and computes rewards based
    on goal-reaching behavior in world coordinates.

    Attributes:
        num_envs: Number of parallel environments.
        obs_dim: Observation vector dimensionality (1061).
        action_shape: Action tensor shape (1, 256).
    """

    # Observation layout
    OBS_DIM = 1061
    # Action shape matches denoiser noise_shape
    ACTION_SHAPE = (1, 256)

    def __init__(
        self,
        args: EnvArgs,
        device: torch.device,
        goal_scheduler_args: GoalSchedulerArgs | None = None,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        """Initialize the RL environment with pretrained models and seed data.

        Loads the MLD checkpoint (denoiser + VAE + diffusion), CLIP text encoder,
        dataset mean/std, and seed motion data. Validates all paths at construction.

        Args:
            args: Environment configuration.
            device: Compute device for all tensors and models.
            goal_scheduler_args: Goal scheduler config. Uses defaults if None.
            reward_weights: Reward function weights. Uses defaults if None.

        Raises:
            FileNotFoundError: If checkpoint, seed data, or mean/std paths are invalid.
            ValueError: If seed data has no valid files or text list is empty.
        """
        self.args = args
        self.device = device
        self.num_envs = args.num_envs
        self.obs_dim = self.OBS_DIM

        # --- Validate locomotion text list (fast check first) ---
        if not args.texts or len(args.texts) == 0:
            raise ValueError(
                "Locomotion text list must contain at least one entry. "
                "Got an empty list."
            )

        # --- Validate paths ---
        self._validate_paths(args)

        # --- Load MLD checkpoint and config ---
        checkpoint_path = Path(args.checkpoint_path)
        mld_ckpt = torch.load(checkpoint_path, map_location="cpu")
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"MLD config.json not found at expected path: {config_path}"
            )
        mld_config = json.loads(config_path.read_text())

        # Store architecture params for reference
        self.history_length = mld_config["history_length"]
        self.future_length = mld_config["future_length"]
        self.feature_dim = mld_config["feature_dim"]
        self.latent_size = mld_config["latent_size"]
        self.latent_width = mld_config["latent_width"]
        self.diffusion_steps = mld_config["diffusion_steps"]
        self.action_shape = (self.latent_size, self.latent_width)
        self.action_dim = self.latent_size * self.latent_width

        # --- Load VAE ---
        vae_ckpt_path = Path(mld_ckpt["vae_checkpoint"])
        if not vae_ckpt_path.exists():
            raise FileNotFoundError(
                f"VAE checkpoint path from MLD checkpoint does not exist: {vae_ckpt_path}"
            )
        vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
        vae_config_path = vae_ckpt_path.parent / "config.json"
        vae_config = json.loads(vae_config_path.read_text()) if vae_config_path.exists() else {}
        # Use EMA weights if available, otherwise regular weights
        vae_state_key = "ema_model_state_dict" if "ema_model_state_dict" in vae_ckpt else "model_state"
        self.vae = AutoMldVae(
            nfeats=self.feature_dim,
            latent_dim=(self.latent_size, self.latent_width),
            h_dim=vae_config.get("h_dim", mld_config.get("h_dim", 256)),
            ff_size=vae_config.get("ff_size", mld_config.get("ff_size", 1024)),
            num_layers=vae_config.get("num_layers", mld_config.get("num_layers", 7)),
            num_heads=vae_config.get("num_heads", mld_config.get("num_heads", 4)),
            dropout=vae_config.get("dropout", mld_config.get("dropout", 0.1)),
        ).to(device)
        self.vae.load_state_dict(vae_ckpt[vae_state_key])
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # --- Load Denoiser ---
        noise_shape = (self.latent_size, self.latent_width)
        history_shape = (self.history_length, self.feature_dim)
        denoiser_type = mld_config.get("denoiser_type", "transformer")

        if denoiser_type == "mlp":
            denoiser = DenoiserMLP(
                h_dim=mld_config.get("denoiser_h_dim", 512),
                n_blocks=mld_config.get("denoiser_blocks", 2),
                clip_dim=mld_config.get("text_dim", 512),
                history_shape=history_shape,
                noise_shape=noise_shape,
                cond_mask_prob=0.0,  # No masking at inference
            ).to(device)
        else:
            denoiser = DenoiserTransformer(
                h_dim=mld_config.get("denoiser_h_dim", 512),
                ff_size=mld_config.get("denoiser_ff_size", 1024),
                num_layers=mld_config.get("denoiser_layers", 8),
                num_heads=mld_config.get("denoiser_heads", 4),
                clip_dim=mld_config.get("text_dim", 512),
                history_shape=history_shape,
                noise_shape=noise_shape,
                cond_mask_prob=0.0,  # No masking at inference
            ).to(device)

        # Use EMA denoiser weights if available
        denoiser_state_key = (
            "ema_denoiser_state_dict"
            if "ema_denoiser_state_dict" in mld_ckpt
            else "denoiser_state"
        )
        denoiser.load_state_dict(mld_ckpt[denoiser_state_key])
        denoiser.eval()
        for param in denoiser.parameters():
            param.requires_grad = False

        # Wrap with classifier-free guidance
        self.denoiser = ClassifierFreeGuidanceWrapper(
            denoiser, guidance_scale=args.guidance_scale
        ).to(device)
        self.denoiser.eval()

        # --- Create diffusion schedule ---
        self.diffusion = GaussianDiffusion(num_steps=self.diffusion_steps)

        # --- Load and freeze CLIP text encoder ---
        clip_version = mld_config.get("clip_version", "ViT-B/32")
        self.clip_model = load_and_freeze_clip(clip_version, device=device)

        # --- Load mean/std arrays ---
        data_root = Path(args.data_root)
        mean_path = data_root / "mean_std" / "Mean.npy"
        std_path = data_root / "mean_std" / "Std.npy"
        self.mean = torch.from_numpy(np.load(mean_path)).float().to(device)  # [272]
        self.std = torch.from_numpy(np.load(std_path)).float().clamp(min=1e-6).to(device)  # [272]
        assert self.mean.shape == (272,), f"Expected mean shape (272,), got {self.mean.shape}"
        assert self.std.shape == (272,), f"Expected std shape (272,), got {self.std.shape}"

        # --- Load seed data ---
        self.seed_data = self._load_seed_data(args.seed_data_path)

        # --- Initialize goal scheduler ---
        if goal_scheduler_args is None:
            goal_scheduler_args = GoalSchedulerArgs()
        self.goal_scheduler = GoalScheduler(goal_scheduler_args)

        # --- Initialize reward weights ---
        if reward_weights is None:
            reward_weights = RewardWeights()
        self.reward_weights = reward_weights

        # --- Pre-encode locomotion texts ---
        self.locomotion_texts = args.texts
        with torch.no_grad():
            self.locomotion_embeddings = encode_text(
                self.clip_model, self.locomotion_texts
            ).to(device)  # [num_texts, 512]

        # --- Allocate batch-dimension state tensors ---
        B = self.num_envs
        self.motion_history = torch.zeros(B, self.history_length, self.feature_dim, device=device)
        self.motion_history_denorm = torch.zeros(B, self.history_length, self.feature_dim, device=device)
        self.transf_rotmat = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).contiguous().clone()
        self.transf_transl = torch.zeros(B, 3, device=device)
        self.goal_location = torch.zeros(B, 3, device=device)
        self.goal_text: list[str] = [""] * B
        self.goal_text_embedding = torch.zeros(B, 512, device=device)
        self.steps = torch.zeros(B, dtype=torch.long, device=device)
        self.prev_pelvis_pos = torch.zeros(B, 3, device=device)
        self.history_world_joints = torch.zeros(B, self.history_length, 22, 3, device=device)
        self.floor_height = torch.zeros(B, device=device)
        self.global_iteration = 0
        self.global_step = 0
        self.export_dir = Path(args.export_dir) if args.export_dir else None
        self.sequences: list[dict] = [self._new_rollout_buffer() for _ in range(B)]

    def _validate_paths(self, args: EnvArgs) -> None:
        """Validate that all required paths exist.

        Raises:
            FileNotFoundError: If any required path does not exist.
        """
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MLD checkpoint path does not exist: {checkpoint_path}"
            )

        seed_data_path = Path(args.seed_data_path)
        if not seed_data_path.exists():
            raise FileNotFoundError(
                f"Seed data path does not exist: {seed_data_path}"
            )

        data_root = Path(args.data_root)
        mean_path = data_root / "mean_std" / "Mean.npy"
        std_path = data_root / "mean_std" / "Std.npy"
        if not mean_path.exists():
            raise FileNotFoundError(
                f"Mean array not found at expected path: {mean_path}"
            )
        if not std_path.exists():
            raise FileNotFoundError(
                f"Std array not found at expected path: {std_path}"
            )

    def _load_seed_data(self, seed_data_path: str) -> list[torch.Tensor]:
        """Load seed motion data from .npy files in the configured directory.

        Each file must have shape (T, 272) with T >= 2.

        Args:
            seed_data_path: Path to directory containing .npy seed files.

        Returns:
            List of normalized seed motion tensors on device.

        Raises:
            ValueError: If no valid seed files are found.
        """
        seed_dir = Path(seed_data_path)
        seed_files = sorted(seed_dir.glob("*.npy"))
        seeds: list[torch.Tensor] = []

        for f in seed_files:
            data = np.load(f)
            if data.ndim != 2 or data.shape[1] != 272:
                continue
            if data.shape[0] < 2:
                continue
            # Store as normalized tensor (for denoiser conditioning)
            motion_tensor = torch.from_numpy(data).float().to(self.device)
            # Normalize
            motion_normalized = (motion_tensor - self.mean) / self.std
            seeds.append(motion_normalized)

        if len(seeds) == 0:
            raise ValueError(
                f"No valid seed data files found at {seed_data_path}. "
                f"Expected .npy files with shape (T, 272) where T >= 2."
            )

        return seeds

    def _new_rollout_buffer(self) -> dict:
        return {
            "motion": [],
            "world_joints": [],
            "transf_rotmat": [],
            "transf_transl": [],
            "goal_location": [],
            "goal_text": [],
            "action": [],
            "obs": [],
        }

    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize motion using dataset mean/std."""
        return (motion - self.mean) / self.std

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        """Denormalize motion using dataset mean/std."""
        return motion * self.std + self.mean

    # ------------------------------------------------------------------
    # Core environment methods
    # ------------------------------------------------------------------

    def reset(
        self,
        batch_idx: torch.Tensor | None = None,
        goal_location: torch.Tensor | None = None,
        goal_texts: list[str] | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Reset specified environments (or all). Returns observation [B, 1061].

        For each environment being reset:
        1. Sample a random seed motion and random start index (ensuring 2 frames remain)
        2. Set motion_history to those 2 consecutive frames (normalized)
        3. Set motion_history_denorm to the denormalized version
        4. Set transf_rotmat to identity [3,3]
        5. Calibrate transf_transl Y so lowest foot joint touches Y=0
        6. Sample a random goal via GoalScheduler
        7. Sample a random locomotion text and cache its CLIP embedding

        Args:
            batch_idx: Optional tensor of environment indices to reset.
                If None, resets all environments.

        Returns:
            Observation tensor of shape (num_envs, 1061).
        """
        if batch_idx is None:
            batch_idx = torch.arange(self.num_envs, device=self.device)
        elif batch_idx.numel() == 0:
            return self.get_observation()

        num_reset = batch_idx.shape[0]

        for i in range(num_reset):
            idx = batch_idx[i].item()

            # 1. Sample a random seed and random start index
            seed_idx = torch.randint(len(self.seed_data), (1,)).item()
            seed_motion = self.seed_data[seed_idx]  # [T, 272] normalized
            T = seed_motion.shape[0]
            # Ensure at least 2 frames remain from start_index
            max_start = T - 2  # inclusive
            start_idx = torch.randint(max_start + 1, (1,)).item()

            # 2. Set motion history (normalized) — 2 consecutive frames
            self.motion_history[idx] = seed_motion[start_idx : start_idx + 2]

            # 3. Set denormalized motion history
            self.motion_history_denorm[idx] = self.denormalize(
                self.motion_history[idx]
            )

            # 4. Set transf_rotmat to identity
            self.transf_rotmat[idx] = torch.eye(3, device=self.device)

            # 5. Calibrate floor height so lowest foot joint Y = 0
            # Use local_to_world_joints on the 2-frame segment with identity transform
            # The function accumulates heading from the data itself, starting at origin
            segment_denorm = self.motion_history_denorm[idx].unsqueeze(0)  # [1, 2, 272]
            with torch.no_grad():
                local_joints = local_to_world_joints(segment_denorm)  # [1, 2, 22, 3]

            # Extract foot joints (indices 7, 8, 10, 11) from first frame
            foot_indices = [7, 8, 10, 11]
            foot_joints_frame0 = local_joints[0, 0, foot_indices, :]  # [4, 3]
            min_foot_y = foot_joints_frame0[:, 1].min().item()

            # Set transf_transl so that min foot Y in world = 0
            # world_y = local_y + transf_transl_y => 0 = min_foot_y + offset
            # offset = -min_foot_y
            self.transf_transl[idx] = torch.tensor(
                [0.0, -min_foot_y, 0.0], device=self.device
            )
            self.floor_height[idx] = -min_foot_y

            # 6. Reset step counter
            self.steps[idx] = 0

            # 7. Compute initial pelvis world position for prev_pelvis_pos
            pelvis_world = local_joints[0, 0, 0, :]  # [3]
            pelvis_world = pelvis_world + self.transf_transl[idx]
            self.prev_pelvis_pos[idx] = pelvis_world
            self.history_world_joints[idx] = local_joints[0] + self.transf_transl[idx].view(1, 1, 3)
            self.sequences[idx] = self._new_rollout_buffer()
            self.sequences[idx]["motion"].append(self.motion_history_denorm[idx].detach().cpu())
            self.sequences[idx]["world_joints"].append(self.history_world_joints[idx].detach().cpu())
            self.sequences[idx]["transf_rotmat"].append(
                self.transf_rotmat[idx].unsqueeze(0).repeat(self.history_length, 1, 1).detach().cpu()
            )
            self.sequences[idx]["transf_transl"].append(
                self.transf_transl[idx].unsqueeze(0).repeat(self.history_length, 1).detach().cpu()
            )

        if goal_location is not None:
            goals = torch.as_tensor(goal_location, dtype=torch.float32, device=self.device)
            if goals.ndim == 1:
                goals = goals.unsqueeze(0)
            if goals.shape[0] != num_reset:
                goals = goals[batch_idx]
            goals = self._normalize_goal_coordinates(goals)
            self.goal_location[batch_idx] = goals
        else:
            self._assign_random_goals(batch_idx, reset_text=False)

        if goal_texts is not None:
            self._set_goal_texts(batch_idx, goal_texts)
        else:
            self._assign_random_texts(batch_idx)

        return self.get_observation()

    def step(
        self,
        action: torch.Tensor,
        next_goal_location: torch.Tensor | None = None,
        next_goal_texts: list[str] | np.ndarray | None = None,
        reset_text: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one motion primitive step.

        Pipeline:
        1. Validate and reshape action to (B, 1, 256)
        2. Run DDIM sampling with action as initial noise and CFG denoiser
        3. Decode latent through VAE to produce 8 frames of 272-dim motion
        4. Denormalize motion using dataset mean/std
        5. Update world transform (heading rotation + root displacement)
        6. Extract world-space joints for reward computation
        7. Compute reward, check termination/truncation
        8. Update motion history for next step conditioning
        9. Auto-reset terminated/truncated environments

        Args:
            action: Policy action tensor. Must have exactly B * 256 elements
                where B = num_envs.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info):
                - observation: [B, 1061] observation vector
                - reward: [B] scalar reward per environment
                - terminated: [B] bool, True if pelvis-goal distance > threshold
                - truncated: [B] bool, True if step count >= max_steps
                - info: dict with 'pelvis_pos' [B, 3], 'step_count' [B],
                  'success_mask' [B], 'reward_info' dict

        Raises:
            ValueError: If action element count != num_envs * 256.
        """
        B = self.num_envs
        device = self.device

        # --- 1. Validate action ---
        expected_numel = B * self.action_dim
        if action.numel() != expected_numel:
            raise ValueError(
                f"Action tensor must have {expected_numel} elements "
                f"(num_envs={B} × {self.action_dim}), got {action.numel()}"
            )

        # Reshape action to denoiser noise shape: (B, 1, 256)
        action = action.to(device=device, dtype=self.motion_history.dtype)
        noise = action.reshape(B, self.latent_size, self.latent_width)
        obs_before_step = self.get_observation().detach()

        # --- 2. DDIM sampling with action as initial noise ---
        # Build model_kwargs for the denoiser (CFG wrapper)
        model_kwargs = {
            "y": {
                "text_embedding": self.goal_text_embedding,  # [B, 512]
                "history_motion_normalized": self.motion_history,  # [B, 2, 272]
            }
        }

        # Run DDIM loop manually with action as initial x_t
        ddim_steps = self.args.ddim_steps
        step_indices = torch.linspace(
            0, self.diffusion.num_steps - 1, ddim_steps, dtype=torch.long, device=device
        )
        x_t = noise  # Use policy action as initial noise
        with torch.no_grad():
            for i in reversed(range(ddim_steps)):
                t = torch.full(
                    (B,), step_indices[i].item(), device=device, dtype=torch.long
                )
                t_prev = torch.full(
                    (B,),
                    step_indices[i - 1].item() if i > 0 else -1,
                    device=device,
                    dtype=torch.long,
                )
                x_t = self.diffusion.ddim_sample(
                    self.denoiser, x_t, t, t_prev, model_kwargs
                )

        # x_t is now the denoised latent z: [B, 1, 256]
        z = x_t

        # --- 3. VAE decode to produce 8 future frames ---
        with torch.no_grad():
            # VAE decode expects z in seq-first format: [latent_size, B, latent_width]
            z_seq_first = z.permute(1, 0, 2)  # [1, B, 256]
            motion_norm = self.vae.decode(
                z_seq_first,
                history_motion=self.motion_history,  # [B, 2, 272]
                nfuture=self.future_length,  # 8
                scale_latent=True,
            )  # [B, 8, 272]

        # --- 4. Denormalize ---
        motion_denorm = self.denormalize(motion_norm)  # [B, 8, 272]

        # --- 5. Extract world-space joints using the transform at segment start ---
        start_rotmat = self.transf_rotmat.clone()
        start_transl = self.transf_transl.clone()
        history_world_joints = self.history_world_joints.clone()
        prev_motion_frame = self.motion_history_denorm[:, -1, :].clone()

        # Extract heading deltas and root velocity from the generated frames
        heading_6d = motion_denorm[:, :, 2:8]  # [B, 8, 6]
        root_vel_xz = motion_denorm[:, :, 0:2]  # [B, 8, 2]

        # Compute cumulative heading rotation for this segment
        segment_headings = cumulative_rotation_batched(heading_6d)  # [B, 8, 3, 3]
        world_joints = self.get_world_joints(
            motion_denorm, transf_rotmat=start_rotmat, transf_transl=start_transl
        )  # [B, 8, 22, 3]

        # Accumulate root XZ displacement in world frame
        # Convert local XZ velocity to 3D
        root_vel_3d = torch.zeros(B, self.future_length, 3, device=device)
        root_vel_3d[:, :, 0] = root_vel_xz[:, :, 0]  # X
        root_vel_3d[:, :, 2] = root_vel_xz[:, :, 1]  # Z

        # Rotate local velocity by segment heading to get segment-local world velocity
        segment_world_vel = torch.einsum(
            "btij,btj->bti", segment_headings, root_vel_3d
        )  # [B, 8, 3]

        # Total displacement in segment-local frame
        segment_displacement = segment_world_vel.sum(dim=1)  # [B, 3]
        # Zero out Y component (root height comes from joint data)
        segment_displacement[:, 1] = 0.0

        # Transform segment displacement to global world frame using current transf_rotmat
        world_displacement = torch.einsum(
            "bij,bj->bi", start_rotmat, segment_displacement
        )  # [B, 3]

        # --- 7. Compute reward ---
        total_reward, success_mask, reward_info = compute_reward(
            world_joints=world_joints,
            prev_pelvis_pos=self.prev_pelvis_pos,
            goal_location=self.goal_location,
            goal_texts=self.goal_text,
            weights=self.reward_weights,
            success_threshold=self.args.success_threshold,
            history_world_joints=history_world_joints,
            prev_motion_frame=prev_motion_frame,
            future_motion=motion_denorm,
        )

        # --- 8. Update state ---
        # Update motion history with last 2 frames (normalized)
        self.motion_history = motion_norm[:, -2:, :].clone()  # [B, 2, 272]
        self.motion_history_denorm = motion_denorm[:, -2:, :].clone()  # [B, 2, 272]
        self.history_world_joints = world_joints[:, -self.history_length :, :, :].clone()

        # Update prev_pelvis_pos with last frame pelvis world position
        self.prev_pelvis_pos = world_joints[:, -1, 0, :].clone()  # [B, 3]

        # Update global transform for the next primitive.
        segment_final_heading = segment_headings[:, -1]  # [B, 3, 3]
        self.transf_transl = start_transl + world_displacement
        self.transf_rotmat = torch.bmm(start_rotmat, segment_final_heading)

        # Increment step counter
        self.steps = self.steps + 1
        self.global_step += B

        if self.args.enable_export:
            self._record_step(
                action=action.reshape(B, self.action_dim),
                obs=obs_before_step,
                motion_denorm=motion_denorm,
                world_joints=world_joints,
                segment_headings=segment_headings,
                start_rotmat=start_rotmat,
            )

        # --- 9. Check termination and truncation ---
        # Termination: pelvis-to-goal XY distance > terminate_threshold
        pelvis_xy = self.prev_pelvis_pos[:, [0, 2]]  # [B, 2]
        goal_xy = self.goal_location[:, [0, 2]]  # [B, 2]
        pelvis_goal_dist = torch.norm(pelvis_xy - goal_xy, dim=-1)  # [B]
        terminated = pelvis_goal_dist > self.args.terminate_threshold

        # Truncation: step count >= max_steps
        truncated = self.steps >= self.args.max_steps

        # Build info dict
        info = {
            "pelvis_pos": self.prev_pelvis_pos.clone(),
            "step_count": self.steps.clone(),
            "success_mask": success_mask,
            "num_success": int(success_mask.sum().item()),
            "num_terminated": int(terminated.sum().item()),
            "num_truncated": int(truncated.sum().item()),
            "reward_info": reward_info,
            "reward_dict": reward_info,
        }

        # --- 10. Assign a new goal when the current one is reached ---
        self._handle_success_goals(
            success_mask,
            next_goal_location=next_goal_location,
            next_goal_texts=next_goal_texts,
            reset_text=reset_text,
        )

        # --- 11. Auto-reset terminated or truncated environments ---
        done_mask = terminated | truncated
        if done_mask.any():
            done_indices = torch.where(done_mask)[0]
            info["final_observation"] = self.get_observation().clone()
            if (
                self.args.enable_export
                and self.export_dir is not None
                and self.global_iteration % self.args.export_interval == 0
            ):
                self.save_rollouts(done_indices)
            self.reset(done_indices)

        # Get observation (after potential resets)
        obs = self.get_observation()

        return obs, total_reward, terminated, truncated, info

    def get_observation(self) -> torch.Tensor:
        """Construct observation vector [B, 1061] from current state.

        The observation is concatenated in fixed order:
            goal_dir (3) + goal_dist (1) + text_embedding (512) +
            motion_history (544) + floor_height (1) = 1061

        Goal direction is computed in the local (heading-removed) frame:
        1. Compute world-frame direction from pelvis to goal on XZ plane
        2. Transform to local frame via transf_rotmat.T (inverse heading rotation)
        3. If angle to forward direction [1, 0, 0] exceeds obs_goal_angle_clip,
           project onto the cone boundary while preserving unit norm

        Returns:
            Observation tensor of shape (num_envs, 1061) with all finite values.
        """
        B = self.num_envs
        device = self.device

        # --- Goal direction in local frame ---
        # World-frame vector from pelvis to goal, projected to XZ plane
        goal_vec_world = self.goal_location - self.prev_pelvis_pos  # [B, 3]
        goal_vec_world[:, 1] = 0.0  # Zero out Y (project to XZ plane)

        # Compute XZ-plane distance
        goal_dist_xz = torch.norm(goal_vec_world[:, [0, 2]], dim=-1, keepdim=False)  # [B]

        # Normalize to unit vector (handle zero-distance case)
        goal_dir_world = goal_vec_world.clone()
        nonzero_mask = goal_dist_xz > 1e-8
        goal_dir_world[nonzero_mask] = (
            goal_dir_world[nonzero_mask]
            / goal_dist_xz[nonzero_mask].unsqueeze(-1)
        )
        # For zero-distance, default to forward direction in world frame
        if (~nonzero_mask).any():
            goal_dir_world[~nonzero_mask] = (
                self.transf_rotmat[~nonzero_mask] @ torch.tensor([1.0, 0.0, 0.0], device=device)
            )

        # Transform to local frame: local_dir = R^T @ world_dir
        # transf_rotmat: [B, 3, 3], goal_dir_world: [B, 3]
        local_goal_dir = torch.einsum(
            "bij,bj->bi", self.transf_rotmat.transpose(1, 2), goal_dir_world
        )  # [B, 3]

        # --- Cone clamping ---
        # Forward direction in local frame is [1, 0, 0] (X-axis, consistent with
        # the heading rotation convention where identity heading = facing +X)
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=device)  # [3]
        max_angle_rad = torch.deg2rad(
            torch.tensor(self.args.obs_goal_angle_clip, device=device)
        )

        # Compute angle between local_goal_dir and forward [1, 0, 0]
        # cos(angle) = dot(local_goal_dir, forward) / (||local_goal_dir|| * ||forward||)
        # Since both should be unit vectors: cos(angle) = local_goal_dir[..., 0]
        cos_angle = local_goal_dir[:, 0].clamp(-1.0, 1.0)  # [B]
        angle = torch.acos(cos_angle)  # [B]

        # Identify directions that exceed the maximum angle
        needs_clamping = angle > max_angle_rad

        if needs_clamping.any():
            # For directions exceeding the cone, project onto cone boundary
            # Decompose local_goal_dir into forward component and lateral component
            # forward component: local_goal_dir[:, 0] * [1, 0, 0]
            # lateral component: local_goal_dir - forward_component (lies in YZ plane of local frame)
            clamped_dirs = local_goal_dir.clone()
            idx = needs_clamping

            # Lateral component (perpendicular to forward [1,0,0] in local frame)
            lateral = clamped_dirs[idx].clone()
            lateral[:, 0] = 0.0  # Remove forward (X) component

            lateral_norm = torch.norm(lateral, dim=-1, keepdim=True)

            # Handle degenerate case: goal exactly along forward axis (lateral = 0)
            # Pick arbitrary perpendicular direction [0, 0, 1]
            degenerate = (lateral_norm.squeeze(-1) < 1e-8)
            if degenerate.any():
                lateral[degenerate] = torch.tensor([0.0, 0.0, 1.0], device=device)
                lateral_norm[degenerate] = 1.0

            lateral_unit = lateral / lateral_norm.clamp(min=1e-8)

            # Reconstruct direction at the cone boundary angle
            # new_dir = sin(max_angle) * lateral_unit + cos(max_angle) * forward
            cos_max = torch.cos(max_angle_rad)
            sin_max = torch.sin(max_angle_rad)

            new_dir = sin_max * lateral_unit + cos_max * forward_local.unsqueeze(0)
            # Normalize to ensure unit norm
            new_dir = new_dir / torch.norm(new_dir, dim=-1, keepdim=True).clamp(min=1e-8)

            clamped_dirs[idx] = new_dir
            local_goal_dir = clamped_dirs

        # Ensure unit norm (defensive)
        local_goal_dir = local_goal_dir / torch.norm(
            local_goal_dir, dim=-1, keepdim=True
        ).clamp(min=1e-8)

        # --- Goal distance (clipped) ---
        goal_dist_clipped = goal_dist_xz.clamp(0.0, self.args.obs_goal_dist_clip)  # [B]
        goal_dist_obs = goal_dist_clipped.unsqueeze(-1)  # [B, 1]

        # --- CLIP text embedding ---
        text_obs = self.goal_text_embedding  # [B, 512]

        # --- Motion history (denormalized, flattened) ---
        motion_obs = self.motion_history_denorm.reshape(B, -1)  # [B, 2*272] = [B, 544]

        # --- Floor height ---
        floor_obs = self.floor_height.unsqueeze(-1)  # [B, 1]

        # --- Concatenate in fixed order ---
        obs = torch.cat(
            [
                local_goal_dir,   # [B, 3]
                goal_dist_obs,    # [B, 1]
                text_obs,         # [B, 512]
                motion_obs,       # [B, 544]
                floor_obs,        # [B, 1]
            ],
            dim=-1,
        )  # [B, 1061]

        # Defensive: replace any NaN/Inf with 0 and log warning
        if not torch.isfinite(obs).all():
            import warnings
            warnings.warn(
                "Non-finite values detected in observation vector. Replacing with 0."
            )
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def get_world_joints(
        self,
        motion_denorm: torch.Tensor,
        transf_rotmat: torch.Tensor | None = None,
        transf_transl: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get world-space joints from denormalized motion using current transform.

        Applies the local_to_world_joints function to get joints in the segment's
        local frame, then transforms them to the global world frame using the
        current transf_rotmat and transf_transl.

        The segment-local joints from local_to_world_joints already account for
        heading rotation and root displacement within the segment. We then apply
        the accumulated world transform to place them in the global coordinate
        system.

        Args:
            motion_denorm: Denormalized motion tensor [B, T, 272].

        Returns:
            World-space joint positions [B, T, 22, 3].
        """
        # Get joints in segment-local world frame (starting from segment origin)
        with torch.no_grad():
            segment_joints = local_to_world_joints(motion_denorm)  # [B, T, 22, 3]

        if transf_rotmat is None:
            transf_rotmat = self.transf_rotmat
        if transf_transl is None:
            transf_transl = self.transf_transl

        # Transform to global world frame:
        # world_joints = transf_rotmat @ segment_joints + transf_transl
        # transf_rotmat: [B, 3, 3], segment_joints: [B, T, 22, 3]
        B, T, J, _ = segment_joints.shape

        # Rotate: apply transf_rotmat to each joint position
        # Reshape for batch matmul: [B, T*22, 3]
        joints_flat = segment_joints.reshape(B, T * J, 3)
        # [B, 3, 3] @ [B, 3, T*22] -> [B, 3, T*22] -> [B, T*22, 3]
        rotated = torch.einsum("bij,bnj->bni", transf_rotmat, joints_flat)

        # Translate: add transf_transl [B, 3] to each joint
        world_joints = rotated + transf_transl.unsqueeze(1)  # [B, T*22, 3]

        # Reshape back to [B, T, 22, 3]
        world_joints = world_joints.reshape(B, T, J, 3)

        return world_joints

    def _assign_random_goals(self, batch_idx: torch.Tensor, reset_text: bool = True) -> None:
        count = batch_idx.numel()
        pelvis_positions = self.prev_pelvis_pos[batch_idx]
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        forward_dir = torch.einsum(
            "bij,j->bi", self.transf_rotmat[batch_idx], local_forward
        )
        forward_dir[:, 1] = 0.0
        forward_dir = forward_dir / torch.norm(forward_dir, dim=-1, keepdim=True).clamp(min=1e-8)
        self.goal_location[batch_idx] = self.goal_scheduler.sample_goal(
            pelvis_positions, forward_dir, count
        )
        if reset_text:
            self._assign_random_texts(batch_idx)

    def _handle_success_goals(
        self,
        success_mask: torch.Tensor,
        next_goal_location: torch.Tensor | None = None,
        next_goal_texts: list[str] | np.ndarray | None = None,
        reset_text: bool = True,
    ) -> None:
        """Apply DART-main goal reset semantics for successful environments."""
        if not success_mask.any():
            return
        success_idx = torch.where(success_mask)[0]
        if next_goal_location is not None and next_goal_texts is not None:
            next_goals = torch.as_tensor(
                next_goal_location, dtype=torch.float32, device=self.device
            )
            if next_goals.ndim == 1:
                next_goals = next_goals.unsqueeze(0)
            if next_goals.shape[0] != success_idx.numel():
                next_goals = next_goals[success_idx]
            self.goal_location[success_idx] = self._normalize_goal_coordinates(next_goals)
            if reset_text:
                self._set_goal_texts(success_idx, next_goal_texts)
        else:
            self._assign_random_goals(success_idx, reset_text=False)

    def curriculum_step(self) -> None:
        """Advance DART-main style goal curriculum and skate-weight curriculum."""
        self.goal_scheduler.curriculum_step()
        self.reward_weights.weight_skate = min(
            self.reward_weights.weight_skate + self.reward_weights.weight_skate_delta,
            self.reward_weights.weight_skate_max,
        )

    def _assign_random_texts(self, batch_idx: torch.Tensor) -> None:
        text_indices = torch.randint(len(self.locomotion_texts), (batch_idx.numel(),))
        for local_i, env_i in enumerate(batch_idx.tolist()):
            text_i = int(text_indices[local_i].item())
            self.goal_text[env_i] = self.locomotion_texts[text_i]
            self.goal_text_embedding[env_i] = self.locomotion_embeddings[text_i]

    def _set_goal_texts(self, batch_idx: torch.Tensor, goal_texts: list[str] | np.ndarray) -> None:
        if isinstance(goal_texts, np.ndarray):
            texts = [str(x) for x in goal_texts.tolist()]
        else:
            texts = [str(x) for x in goal_texts]
        if len(texts) != batch_idx.numel():
            texts = [texts[i] for i in batch_idx.detach().cpu().tolist()]
        embeddings = encode_text(self.clip_model, texts).to(self.device)
        for local_i, env_i in enumerate(batch_idx.tolist()):
            self.goal_text[env_i] = texts[local_i]
            self.goal_text_embedding[env_i] = embeddings[local_i]

    def _normalize_goal_coordinates(self, goals: torch.Tensor) -> torch.Tensor:
        """Accept internal [x, y, z] or 2D horizontal [x, z] goals."""
        goals = goals.to(device=self.device, dtype=torch.float32)
        if goals.shape[-1] == 2:
            converted = torch.zeros(goals.shape[0], 3, device=self.device)
            converted[:, 0] = goals[:, 0]
            converted[:, 2] = goals[:, 1]
            return converted
        if goals.shape[-1] != 3:
            raise ValueError(f"Expected goals with 2 or 3 coordinates, got {goals.shape}")
        return goals

    def _record_step(
        self,
        action: torch.Tensor,
        obs: torch.Tensor,
        motion_denorm: torch.Tensor,
        world_joints: torch.Tensor,
        segment_headings: torch.Tensor,
        start_rotmat: torch.Tensor,
    ) -> None:
        per_frame_rotmat = torch.einsum("bij,btjk->btik", start_rotmat, segment_headings)
        per_frame_transl = world_joints[:, :, 0, :]
        for env_i in range(self.num_envs):
            seq = self.sequences[env_i]
            seq["motion"].append(motion_denorm[env_i].detach().cpu())
            seq["world_joints"].append(world_joints[env_i].detach().cpu())
            seq["transf_rotmat"].append(per_frame_rotmat[env_i].detach().cpu())
            seq["transf_transl"].append(per_frame_transl[env_i].detach().cpu())
            seq["goal_location"].append(self.goal_location[env_i].detach().cpu())
            seq["goal_text"].append(self.goal_text[env_i])
            seq["action"].append(action[env_i].detach().cpu())
            seq["obs"].append(obs[env_i].detach().cpu())

    def save_rollouts(self, batch_idx: torch.Tensor) -> None:
        if self.export_dir is None:
            return
        save_root = self.export_dir / f"iter_{self.global_iteration}"
        save_root.mkdir(parents=True, exist_ok=True)
        for env_i in batch_idx[: self.args.max_export].detach().cpu().tolist():
            seq = self.sequences[env_i]
            if len(seq["action"]) == 0:
                continue
            payload = {
                "motion": torch.cat(seq["motion"], dim=0).numpy(),
                "world_joints": torch.cat(seq["world_joints"], dim=0).numpy()
                if seq["world_joints"]
                else None,
                "transf_rotmat": torch.cat(seq["transf_rotmat"], dim=0).numpy(),
                "transf_transl": torch.cat(seq["transf_transl"], dim=0).numpy(),
                "goal_location": torch.stack(seq["goal_location"], dim=0).numpy()
                if seq["goal_location"]
                else np.zeros((0, 3), dtype=np.float32),
                "goal_text": list(seq["goal_text"]),
                "action": torch.stack(seq["action"], dim=0).numpy(),
                "obs": torch.stack(seq["obs"], dim=0).numpy(),
            }
            path = save_root / f"step_{self.global_step}_env_{env_i}.pkl"
            with path.open("wb") as f:
                pickle.dump(payload, f)
