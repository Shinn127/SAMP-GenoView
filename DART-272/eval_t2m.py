"""
Evaluate DART-272 text-to-motion generation using MotionStreamer's evaluator.

Metrics: FID, Diversity, R-Precision (Top1/2/3), MM-dist (matching score)

Usage:
    python DART-272/eval_t2m.py \
        --checkpoint DART-272/outputs/mld_run2/checkpoint_best.pt \
        --evaluator-dir MotionStreamer/Evaluator_272 \
        --data-root humanml3d_272 \
        --guidance-scale 5.0 \
        --ddim-steps 10

The script:
1. Loads DART-272 model (VAE + Denoiser)
2. Iterates over HumanML3D-272 test set
3. For each text, generates motion via autoregressive rollout
4. Computes FID, Diversity, R-Precision, MM-dist using MotionStreamer's evaluator
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy import linalg
from tqdm import tqdm
import codecs as cs
import random

# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DART-272 with MotionStreamer evaluator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DART-272 MLD checkpoint")
    parser.add_argument("--evaluator-dir", type=str, default="../MotionStreamer/Evaluator_272",
                        help="Path to MotionStreamer Evaluator_272 directory")
    parser.add_argument("--data-root", type=str, default="../humanml3d_272",
                        help="Path to humanml3d_272 data root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--ddim-steps", type=int, default=10,
                        help="DDIM steps (0 = full DDPM)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit number of batches for quick testing (0 = all)")
    return parser.parse_args()


# ============================================================
# Eval dataset (same format as MotionStreamer's dataset_eval_t2m)
# ============================================================

class EvalText2MotionDataset(Dataset):
    """Test set loader that returns (caption, motion, length) like MotionStreamer's eval."""

    def __init__(self, data_root: str, is_test: bool = True, unit_length: int = 4,
                 max_motion_length: int = 300, min_motion_len: int = 60, fps: int = 30):
        self.data_root = Path(data_root)
        self.motion_dir = self.data_root / "motion_data"
        self.text_dir = self.data_root / "texts"
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length

        mean = np.load(self.data_root / "mean_std" / "Mean.npy")
        std = np.load(self.data_root / "mean_std" / "Std.npy")
        self.mean = mean
        self.std = std

        split_file = self.data_root / "split" / ("test.txt" if is_test else "val.txt")
        id_list = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

        self.data_dict = {}
        new_name_list = []
        length_list = []

        for name in id_list:
            motion_path = self.motion_dir / f"{name}.npy"
            text_path = self.text_dir / f"{name}.txt"
            if not motion_path.exists() or not text_path.exists():
                continue
            motion = np.load(motion_path)
            if len(motion) < min_motion_len or len(motion) >= max_motion_length:
                continue

            text_data = []
            flag = False
            with cs.open(str(text_path)) as f:
                for line in f.readlines():
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    f_tag = float(line_split[2]) if line_split[2] != 'nan' else 0.0
                    to_tag = float(line_split[3]) if line_split[3] != 'nan' else 0.0

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(caption)
                    else:
                        n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                        if len(n_motion) < min_motion_len or len(n_motion) >= max_motion_length:
                            continue
                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in self.data_dict:
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        self.data_dict[new_name] = {
                            'motion': n_motion, 'length': len(n_motion), 'text': [caption]
                        }
                        new_name_list.append(new_name)
                        length_list.append(len(n_motion))

            if flag:
                self.data_dict[name] = {
                    'motion': motion, 'length': len(motion), 'text': text_data
                }
                new_name_list.append(name)
                length_list.append(len(motion))

        self.name_list = new_name_list
        self.length_arr = np.array(length_list)
        print(f"Eval dataset: {len(self.name_list)} samples loaded")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        caption = random.choice(text_list)

        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Normalize
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)

        return caption, motion, m_length


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    captions = [b[0] for b in batch]
    motions = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    lengths = torch.tensor([b[2] for b in batch]).long()
    return captions, motions, lengths


# ============================================================
# DART-272 generation
# ============================================================

def load_dart272_model(checkpoint_path: str, device: torch.device):
    """Load DART-272 VAE + Denoiser from checkpoint."""
    from dart272.denoiser import ClassifierFreeGuidanceWrapper, DenoiserTransformer, DenoiserMLP
    from dart272.diffusion import GaussianDiffusion
    from dart272.text import load_and_freeze_clip
    from dart272.vae import AutoMldVae
    from dart272.utils import load_json

    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = load_json(checkpoint_path.parent / "config.json")
    mvae_ckpt_path = Path(ckpt["vae_checkpoint"])
    mvae_cfg = load_json(mvae_ckpt_path.parent / "config.json")
    mvae_ckpt = torch.load(mvae_ckpt_path, map_location="cpu", weights_only=False)

    # Build VAE
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

    # Build denoiser
    history_shape = (cfg["history_length"], cfg["feature_dim"])
    noise_shape = (cfg["latent_size"], cfg["latent_width"])
    if cfg["denoiser_type"] == "mlp":
        denoiser = DenoiserMLP(
            h_dim=cfg["denoiser_h_dim"],
            n_blocks=cfg.get("denoiser_blocks", 2),
            clip_dim=cfg["text_dim"],
            history_shape=history_shape,
            noise_shape=noise_shape,
            cond_mask_prob=0.1,
        )
    else:
        denoiser = DenoiserTransformer(
            h_dim=cfg["denoiser_h_dim"],
            ff_size=cfg.get("denoiser_ff_size", 1024),
            num_layers=cfg["denoiser_layers"],
            num_heads=cfg.get("denoiser_heads", 4),
            clip_dim=cfg["text_dim"],
            history_shape=history_shape,
            noise_shape=noise_shape,
            cond_mask_prob=0.1,
        )
    denoiser = denoiser.to(device)
    denoiser.load_state_dict(ckpt["denoiser_state"])
    denoiser.eval()

    # CLIP
    clip_model = load_and_freeze_clip(cfg.get("clip_version", "ViT-B/32"), device=device)

    # Diffusion
    diffusion = GaussianDiffusion(num_steps=cfg["diffusion_steps"])

    return vae, denoiser, clip_model, diffusion, cfg, mvae_cfg


@torch.no_grad()
def generate_motion_dart272(
    text: str,
    target_length: int,
    vae, denoiser, clip_model, diffusion, cfg,
    dataset_mean: torch.Tensor,
    dataset_std: torch.Tensor,
    device: torch.device,
    guidance_scale: float = 5.0,
    ddim_steps: int = 10,
    seed_history: torch.Tensor | None = None,
):
    """Generate a single motion sequence using DART-272 autoregressive rollout."""
    from dart272.denoiser import ClassifierFreeGuidanceWrapper
    from dart272.text import encode_text

    history_length = cfg["history_length"]
    future_length = cfg["future_length"]

    # Calculate how many primitives we need
    num_primitives = max(1, (target_length - history_length + future_length - 1) // future_length)

    # Wrap denoiser with CFG
    if guidance_scale > 1.0:
        model = ClassifierFreeGuidanceWrapper(denoiser, guidance_scale=guidance_scale)
    else:
        model = denoiser

    # Initialize history (use zeros normalized if no seed)
    if seed_history is not None:
        history = seed_history.unsqueeze(0).to(device)
    else:
        # Use zero-pose as seed (normalized)
        history = torch.zeros(1, history_length, 272, device=device)

    # Normalize mean/std for the generation
    mean = dataset_mean.to(device)
    std = dataset_std.to(device).clamp(min=1e-6)

    generated_frames = []

    # Encode text once (same text for all primitives)
    text_embedding = encode_text(clip_model, [text], force_empty_zero=True)

    for _ in range(num_primitives):
        # Sample latent
        if ddim_steps > 0:
            latent = diffusion.ddim_sample_loop(
                model,
                shape=(1, cfg["latent_size"], cfg["latent_width"]),
                model_kwargs={"y": {
                    "text_embedding": text_embedding,
                    "history_motion_normalized": history,
                }},
                device=device,
                ddim_steps=ddim_steps,
            )
        else:
            latent = diffusion.p_sample_loop(
                model,
                shape=(1, cfg["latent_size"], cfg["latent_width"]),
                model_kwargs={"y": {
                    "text_embedding": text_embedding,
                    "history_motion_normalized": history,
                }},
                device=device,
            )

        # Decode
        latent_perm = latent.permute(1, 0, 2)  # [latent_size, B, width]
        future = vae.decode(latent_perm, history, future_length, scale_latent=True)
        # future: [1, future_length, 272] (normalized)

        generated_frames.append(future.squeeze(0))  # [future_length, 272]

        # Update history
        all_frames = torch.cat([history, future], dim=1)
        history = all_frames[:, -history_length:]

    # Concatenate all generated frames
    motion = torch.cat(generated_frames, dim=0)  # [num_primitives * future_length, 272]

    # Trim to target length
    motion = motion[:target_length]

    return motion  # normalized


# ============================================================
# Evaluation metrics (same as MotionStreamer)
# ============================================================

def euclidean_distance_matrix(matrix1, matrix2):
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(np.clip(d1 + d2 + d3, 0, None))


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ============================================================
# Main evaluation loop
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Add DART-272 to path
    dart_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(dart_root))

    # Load DART-272 model
    print("Loading DART-272 model...")
    vae, denoiser, clip_model, diffusion, cfg, mvae_cfg = load_dart272_model(
        args.checkpoint, device
    )
    print(f"  Denoiser: {cfg['denoiser_type']}, layers={cfg['denoiser_layers']}")
    print(f"  Diffusion steps: {cfg['diffusion_steps']}, DDIM: {args.ddim_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")

    # Load evaluator
    evaluator_dir = Path(args.evaluator_dir).resolve()
    sys.path.insert(0, str(evaluator_dir))
    print(f"Loading evaluator from {evaluator_dir}...")

    from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
    from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    distilbert_path = str(evaluator_dir / "distilbert-base-uncased")
    if not (evaluator_dir / "distilbert-base-uncased").exists():
        distilbert_path = "distilbert-base-uncased"  # fallback to HF auto-download

    textencoder = DistilbertActorAgnosticEncoder(distilbert_path, num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

    evaluator_ckpt_path = evaluator_dir / "epoch=99.ckpt"
    print(f"  Loading evaluator checkpoint: {evaluator_ckpt_path}")
    ckpt = torch.load(evaluator_ckpt_path, map_location="cpu", weights_only=False)

    textencoder_ckpt = {k.replace("textencoder.", ""): v
                        for k, v in ckpt['state_dict'].items() if k.startswith("textencoder.")}
    motionencoder_ckpt = {k.replace("motionencoder.", ""): v
                          for k, v in ckpt['state_dict'].items() if k.startswith("motionencoder.")}
    textencoder.load_state_dict(textencoder_ckpt, strict=True)
    motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
    textencoder.eval().to(device)
    motionencoder.eval().to(device)
    print("  Evaluator loaded.")

    # Load eval dataset
    print(f"Loading eval dataset from {args.data_root}...")
    eval_dataset = EvalText2MotionDataset(args.data_root, is_test=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True
    )

    # Dataset stats for normalization
    dataset_mean = torch.from_numpy(eval_dataset.mean).float()
    dataset_std = torch.from_numpy(eval_dataset.std).float()

    # ---- Evaluation loop ----
    print("\nStarting evaluation...")
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = np.array([0.0, 0.0, 0.0])
    R_precision_pred = np.array([0.0, 0.0, 0.0])
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0

    for batch_idx, (texts, gt_motion, m_lengths) in enumerate(tqdm(eval_loader, desc="Evaluating")):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        bs = len(texts)
        seq_len = gt_motion.shape[1]

        # Generate motion for each sample in batch
        pred_pose_eval = torch.zeros((bs, seq_len, 272), device=device)
        pred_len = torch.ones(bs).long()

        for k in range(bs):
            target_len = m_lengths[k].item()
            gen_motion = generate_motion_dart272(
                text=texts[k],
                target_length=target_len,
                vae=vae, denoiser=denoiser, clip_model=clip_model,
                diffusion=diffusion, cfg=cfg,
                dataset_mean=dataset_mean, dataset_std=dataset_std,
                device=device,
                guidance_scale=args.guidance_scale,
                ddim_steps=args.ddim_steps,
            )
            cur_len = gen_motion.shape[0]
            actual_len = min(cur_len, seq_len)
            pred_len[k] = actual_len
            pred_pose_eval[k, :actual_len] = gen_motion[:actual_len]

        # Encode with evaluator
        with torch.no_grad():
            et_pred = textencoder(texts).loc
            em_pred = motionencoder(pred_pose_eval, pred_len).loc

            gt_motion_dev = gt_motion.float().to(device)
            et = textencoder(texts).loc
            em = motionencoder(gt_motion_dev, m_lengths).loc

        motion_annotation_list.append(em.cpu().numpy())
        motion_pred_list.append(em_pred.cpu().numpy())

        # R-precision
        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match

        temp_R, temp_match = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # ---- Compute final metrics ----
    print(f"\nTotal samples evaluated: {nb_sample}")

    motion_annotation_np = np.concatenate(motion_annotation_list, axis=0)
    motion_pred_np = np.concatenate(motion_pred_list, axis=0)

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity_real = calculate_diversity(motion_annotation_np, min(300, nb_sample // 2))
    diversity_pred = calculate_diversity(motion_pred_np, min(300, nb_sample // 2))

    R_precision_real /= nb_sample
    R_precision_pred /= nb_sample
    matching_score_real /= nb_sample
    matching_score_pred /= nb_sample

    print("\n" + "=" * 60)
    print("DART-272 Evaluation Results")
    print("=" * 60)
    print(f"  FID:              {fid:.4f}")
    print(f"  Diversity (real): {diversity_real:.4f}")
    print(f"  Diversity (pred): {diversity_pred:.4f}")
    print(f"  R-Precision Top1: {R_precision_pred[0]:.4f} (real: {R_precision_real[0]:.4f})")
    print(f"  R-Precision Top2: {R_precision_pred[1]:.4f} (real: {R_precision_real[1]:.4f})")
    print(f"  R-Precision Top3: {R_precision_pred[2]:.4f} (real: {R_precision_real[2]:.4f})")
    print(f"  MM-dist (pred):   {matching_score_pred:.4f} (real: {matching_score_real:.4f})")
    print("=" * 60)
    print(f"\nConfig: guidance={args.guidance_scale}, ddim_steps={args.ddim_steps}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
