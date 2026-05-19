# DART-272

Diffusion-based Autoregressive Motion Model for 272-dim HumanML3D features.  
Independent reproduction of [DartControl (ICLR 2025)](https://arxiv.org/abs/2410.05260) using the 272-dim motion representation.

## Architecture

```
Text (CLIP) + History (2 frames) → Denoiser (Transformer, 8 layers)
    → Latent (1×256) → VAE Decoder (Skip Transformer, 7 layers)
    → Future Motion (8 frames × 272 dim)
    → Autoregressive rollout
```

## Environment

```bash
conda activate mcc
```

## Data Layout

```
../humanml3d_272/
  motion_data/*.npy      # [T, 272] motion sequences
  texts/*.txt            # text annotations per sequence
  split/{train,val,test}.txt
  mean_std/{Mean.npy, Std.npy}
  .cache/clip_embeddings.pt  # precomputed CLIP embeddings (optional)
```

---

## Training

### 0. Precompute CLIP Embeddings (optional, speeds up MLD training)

```bash
python DART-272/precompute_clip.py --data-root humanml3d_272
```

### 1. Train Motion Primitive VAE

```bash
python DART-272/train_mvae.py \
    --data-root humanml3d_272 \
    --save-dir DART-272/outputs/mvae_run2 \
    --stage1-steps 100000 \
    --stage2-steps 50000 \
    --stage3-steps 50000
```

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--h-dim` | 256 | Hidden dimension |
| `--num-layers` | 7 | Transformer layers (must be odd for Skip connections) |
| `--latent-width` | 256 | Latent space width |
| `--kl-weight` | 1e-6 | KL divergence weight |
| `--delta-weight` | 100 | Joints temporal delta loss weight |
| `--transl-delta-weight` | 100 | Root translation delta weight |
| `--orient-delta-weight` | 100 | Heading orientation delta weight |
| `--ema-decay` | 0.999 | EMA model averaging |
| `--batch-size` | 128 | |
| `--lr` | 1e-4 | Learning rate (linearly annealed) |

Resume training:
```bash
python DART-272/train_mvae.py \
    --save-dir DART-272/outputs/mvae_run2 \
    --resume DART-272/outputs/mvae_run2/checkpoint_last.pt \
    --stage1-steps 100000 --stage2-steps 50000 --stage3-steps 50000
```

### 2. Train Latent Diffusion Denoiser (MLD)

```bash
python DART-272/train_mld.py \
    --data-root humanml3d_272 \
    --save-dir DART-272/outputs/mld_run2 \
    --mvae-ckpt DART-272/outputs/mvae_run2/checkpoint_best.pt \
    --stage1-steps 100000 \
    --stage2-steps 100000 \
    --stage3-steps 100000
```

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--denoiser-type` | transformer | Architecture (transformer or mlp) |
| `--denoiser-layers` | 8 | Transformer layers |
| `--denoiser-h-dim` | 512 | Hidden dimension |
| `--batch-size` | 1024 | (reduce if OOM) |
| `--latent-weight` | 1.0 | Latent reconstruction loss |
| `--feature-weight` | 1.0 | Feature reconstruction loss |
| `--delta-weight` | 1e4 | Joints delta consistency |
| `--transl-delta-weight` | 1e4 | Root translation delta |
| `--orient-delta-weight` | 1e4 | Heading orientation delta |
| `--full-rollout` | 1 | Use full DDPM loop for rollout history |
| `--cond-mask-prob` | 0.1 | Classifier-free guidance dropout |
| `--diffusion-steps` | 10 | Diffusion timesteps |
| `--ema-decay` | 0.999 | EMA model averaging |

### 3. Three-Stage Curriculum Learning

Both VAE and MLD use the same curriculum:
- **Stage 1**: Pure reconstruction/denoising, no rollout (`rollout_prob = 0`)
- **Stage 2**: Linearly increasing rollout probability (`0 → 1`)
- **Stage 3**: Full rollout only (`rollout_prob = 1`)

### TensorBoard

```bash
tensorboard --logdir DART-272/outputs/mvae_run2/tb_logs
tensorboard --logdir DART-272/outputs/mld_run2/tb_logs
```

---

## Inference (Rollout)

Generate motion from a text timeline:

```bash
python DART-272/rollout.py \
    --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
    --text-prompt "walk forward*8,turn left*2,walk forward*8" \
    --guidance-scale 5.0 \
    --ddim-steps 10
```

Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--guidance-scale` | 5.0 | CFG scale (1.0 = no guidance) |
| `--ddim-steps` | 0 | DDIM steps (0 = use full DDPM) |
| `--seed` | 0 | Random seed |
| `--seed-motion-path` | None | Custom initial pose (.npy) |

Text prompt format: `action*num_primitives` separated by commas.  
Each primitive = 8 frames at 30fps ≈ 0.27 seconds.

---

## RL Control Policy

Train a PPO policy that outputs 256D diffusion noise actions for goal reaching:

```bash
conda activate mcc
python DART-272/control/train.py \
    --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
    --data-root humanml3d_272 \
    --seed-data-path DART-272/data/rl_seed \
    --save-dir DART-272/outputs/rl_control \
    --num-envs 256 \
    --num-steps 32 \
    --num-iterations 500
```

`control/train.py` auto-creates `DART-272/data/rl_seed` from valid HumanML3D-272 training motions when the seed directory is missing. Each checkpoint stores the policy, optimizer, environment args, policy args, reward weights, and curriculum state.

Evaluate a trained policy on a goal sequence:

```bash
python DART-272/control/test.py \
    --policy-checkpoint DART-272/outputs/rl_control/checkpoint_last.pt \
    --goal-json DART-main/data/test_locomotion/test_walk_long.json \
    --output-dir DART-272/outputs/rl_control/test_rollouts
```

The test export pickle contains the 272D motion sequence, world-space joints, root translations, root orientations, success flags, and goal metadata for GenoView-side inspection.

---

## Optimization-Based Applications

### Motion In-betweening

Generate motion transitioning through multiple goal keyframes (world-space joint matching):

```bash
python DART-272/optim_inbetween.py \
    --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
    --mvae-ckpt DART-272/outputs/mvae_run2/checkpoint_best.pt \
    --start-motion humanml3d_272/motion_data/000962.npy --start-frame 0 \
    --goal-frames \
        humanml3d_272/motion_data/000962.npy:120:120 \
        humanml3d_272/motion_data/000003.npy:50:200 \
        humanml3d_272/motion_data/000962.npy:last:295 \
    --text-prompt "walk forward" \
    --num-primitives 37 \
    --optim-steps 300 \
    --ddim-steps 10 \
    --output-name my_inbetween
```

Goal format: `motion_path:source_frame:gen_frame`
- `motion_path` = .npy file to extract target pose from (each goal can use a different file)
- `source_frame` = frame index in that file (`last` for last frame)
- `gen_frame` = frame index in generated sequence where this pose should appear

Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--goal-frames` | required | One or more goal specs (see format above) |
| `--num-primitives` | 10 | Duration (×8 frames) |
| `--optim-steps` | 300 | Optimization iterations |
| `--lr` | 0.01 | Learning rate |
| `--guidance-scale` | 5.0 | CFG during optimization |
| `--ddim-steps` | 10 | DDIM sampling steps |
| `--unit-grad` | 1 | Normalize gradient (prevents vanishing) |
| `--anneal-lr` | 1 | Linearly decay learning rate |
| `--weight-jerk` | 0.0 | Smoothness loss weight |
| `--weight-floor` | 0.0 | Floor penetration loss weight |
| `--output-dir` | auto | Output directory |
| `--output-name` | auto | Output filename stem (without extension) |

### Joint Trajectory Control

Control specific joints to follow world-space trajectory waypoints:

```bash
python DART-272/optim_trajectory.py \
    --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
    --mvae-ckpt DART-272/outputs/mvae_run2/checkpoint_best.pt \
    --start-motion humanml3d_272/motion_data/000962.npy --start-frame 0 \
    --trajectory DART-272/data/traj_example.json \
    --text-prompt "walk forward" \
    --num-primitives 10 \
    --optim-steps 300 \
    --ddim-steps 10
```

Trajectory JSON format (world coordinates, meters):
```json
{
  "waypoints": [
    {"frame": 20, "joint": 0, "position": [0.5, 0.9, 0.25]},
    {"frame": 40, "joint": 0, "position": [1.0, 0.9, 0.5]},
    {"frame": 60, "joint": 20, "position": [0.4, 1.5, 0.2]}
  ]
}
```

Joint indices (22 joints):
```
0=Pelvis, 1=L_Hip, 2=R_Hip, 3=Spine1, 4=L_Knee, 5=R_Knee,
6=Spine2, 7=L_Ankle, 8=R_Ankle, 9=Spine3, 10=L_Foot, 11=R_Foot,
12=Neck, 13=L_Collar, 14=R_Collar, 15=Head, 16=L_Shoulder,
17=R_Shoulder, 18=L_Elbow, 19=R_Elbow, 20=L_Wrist, 21=R_Wrist
```

All positions in trajectory waypoints are in **world coordinates** (meters).
The optimization uses a differentiable local-to-world transform to accumulate
root velocity and heading, then computes loss in world space.

---

## Quantitative Evaluation

Evaluate text-to-motion generation quality using MotionStreamer's evaluator (FID, Diversity, R-Precision, MM-dist):

```bash
python DART-272/eval_t2m.py \
    --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
    --evaluator-dir MotionStreamer/Evaluator_272 \
    --data-root humanml3d_272 \
    --guidance-scale 5.0 \
    --ddim-steps 10 \
    --batch-size 32
```

Quick test (1 batch only):
```bash
python DART-272/eval_t2m.py \
    --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
    --evaluator-dir MotionStreamer/Evaluator_272 \
    --data-root humanml3d_272 \
    --batch-size 4 --max-batches 1
```

Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | required | DART-272 MLD checkpoint |
| `--evaluator-dir` | `../MotionStreamer/Evaluator_272` | Path to evaluator (needs `epoch=99.ckpt` and `distilbert-base-uncased/`) |
| `--data-root` | `../humanml3d_272` | HumanML3D-272 data root |
| `--guidance-scale` | 5.0 | Classifier-free guidance scale |
| `--ddim-steps` | 10 | DDIM sampling steps (0 = full DDPM) |
| `--batch-size` | 32 | Evaluation batch size |
| `--max-batches` | 0 | Limit batches for quick testing (0 = all) |
| `--device` | auto | Device (auto/cuda/mps/cpu) |

Metrics reported:
| Metric | Description |
|--------|-------------|
| FID | Fréchet Inception Distance between generated and real motion distributions (lower is better) |
| Diversity | Variance of generated motions in latent space |
| R-Precision Top1/2/3 | Text-motion retrieval accuracy |
| MM-dist | Mean matching distance between text and motion embeddings (lower is better) |

Prerequisites:
- `MotionStreamer/Evaluator_272/epoch=99.ckpt` — evaluator checkpoint
- `MotionStreamer/Evaluator_272/distilbert-base-uncased/` — DistilBERT model
- `humanml3d_272/` — motion data, texts, split, mean_std

---

## Visualization

All outputs are `.npy` files with shape `[T, 272]`. Visualize with GenoView:

```bash
# Rollout output
python Genoview/genoview.py --motion DART-272/outputs/mld_run2/rollout/walk_forwardx8.npy

# In-betweening output
python Genoview/genoview.py --motion DART-272/outputs/mld_run2/inbetween/inbetween_walk_forward_p10.npy

# Trajectory output
python Genoview/genoview.py --motion DART-272/outputs/mld_run2/trajectory/traj_walk_forward_p10.npy

# Live inference (real-time text-to-motion)
python Genoview/genoview.py --live DART-272/outputs/mld_run2/checkpoint_best.pt
```

---

## Project Structure

```
DART-272/
├── dart272/
│   ├── data.py              # Dataset loader (272-dim HumanML3D)
│   ├── vae.py               # Skip Transformer VAE
│   ├── denoiser.py          # Transformer/MLP denoiser + CFG wrapper
│   ├── diffusion.py         # Gaussian diffusion (DDPM + DDIM)
│   ├── text.py              # CLIP text encoding
│   ├── world_transform.py   # Differentiable local-to-world transform
│   └── utils.py             # Utilities
├── train_mvae.py            # VAE training
├── train_mld.py             # Denoiser training
├── rollout.py               # Text-conditioned generation
├── eval_t2m.py              # Quantitative evaluation (FID, R-Precision, etc.)
├── optim_inbetween.py       # Motion in-betweening optimization
├── optim_trajectory.py      # Joint trajectory control optimization
├── precompute_clip.py       # CLIP embedding precomputation
├── data/
│   └── traj_example.json        # Example trajectory (world coordinates)
└── outputs/                 # Training checkpoints and results
```

---

## Key Differences from DART-main

| Aspect | DART-main | DART-272 |
|--------|-----------|----------|
| Motion representation | 276-dim SMPL parameters | 272-dim kinematic features |
| Body model | SMPL-X/H (requires body model) | No body model needed |
| Coordinate system | World coords via SMPL FK | Root-relative (+ differentiable world transform) |
| Canonicalization | Explicit facing-direction alignment | Implicit in 272-dim representation |
| FPS | 20 (HML3D) / 30 (BABEL) | 30 |
| Visualization | Pyrender + Blender | GenoView (real-time OpenGL) |

## Not Yet Implemented

- Scene interaction (SDF collision/contact)
