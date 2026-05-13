from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dart272.data import HumanML3D272Dataset
from dart272.denoiser import ClassifierFreeGuidanceWrapper, DenoiserMLP, DenoiserTransformer
from dart272.diffusion import GaussianDiffusion
from dart272.text import encode_text, load_and_freeze_clip
from dart272.utils import ensure_dir, load_json, resolve_device, set_seed
from dart272.vae import AutoMldVae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text-prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed-motion-path", type=str, default=None)
    parser.add_argument("--seed-start-frame", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="Classifier-free guidance scale. 1.0=no guidance, >1 strengthens text control.")
    parser.add_argument("--ddim-steps", type=int, default=0,
                        help="If > 0, use DDIM sampling with this many steps instead of full DDPM.")
    return parser.parse_args()


def parse_timeline(prompt: str) -> list[str]:
    texts: list[str] = []
    for segment in prompt.split(","):
        segment = segment.strip()
        if not segment:
            continue
        text, count = segment.rsplit("*", 1)
        texts.extend([text.strip()] * int(count))
    return texts


def build_denoiser(cfg: dict, cond_mask_prob: float = 0.0) -> torch.nn.Module:
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_json(checkpoint_path.parent / "config.json")
    mvae_cfg = load_json(Path(ckpt["vae_checkpoint"]).parent / "config.json")
    mvae_ckpt = torch.load(ckpt["vae_checkpoint"], map_location="cpu")

    dataset = HumanML3D272Dataset(
        data_root=cfg["data_root"],
        split="test",
        history_length=cfg["history_length"],
        future_length=cfg["future_length"],
        num_primitives=cfg["num_primitives"],
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

    clip_model = load_and_freeze_clip(cfg.get("clip_version", "ViT-B/32"), device=device)

    # Build denoiser — keep cond_mask_prob > 0 so the mask_cond logic is available
    # for the guidance wrapper's uncond pass.
    denoiser = build_denoiser(cfg, cond_mask_prob=0.1).to(device)
    denoiser.load_state_dict(ckpt["denoiser_state"])
    denoiser.eval()

    # Wrap with classifier-free guidance if scale > 1
    if args.guidance_scale > 1.0:
        model = ClassifierFreeGuidanceWrapper(denoiser, guidance_scale=args.guidance_scale)
        print(f"  Using classifier-free guidance with scale={args.guidance_scale}")
    else:
        model = denoiser

    diffusion = GaussianDiffusion(num_steps=cfg["diffusion_steps"])
    texts = parse_timeline(args.text_prompt)

    if args.seed_motion_path is not None:
        seed_motion = torch.from_numpy(np.load(args.seed_motion_path)).float()
        start = args.seed_start_frame
        history = seed_motion[start : start + cfg["history_length"]]
    else:
        sample = dataset[0]
        history = dataset.denormalize(sample["motion"][: cfg["history_length"]])

    history = dataset.normalize(history).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = [dataset.denormalize(history.squeeze(0)).detach().cpu()]
        for text in texts:
            text_embedding = encode_text(clip_model, [text], force_empty_zero=True)
            latent = diffusion.p_sample_loop(
                model,
                shape=(1, cfg["latent_size"], cfg["latent_width"]),
                model_kwargs={
                    "y": {
                        "text_embedding": text_embedding,
                        "history_motion_normalized": history,
                    }
                },
                device=device,
            ) if args.ddim_steps <= 0 else diffusion.ddim_sample_loop(
                model,
                shape=(1, cfg["latent_size"], cfg["latent_width"]),
                model_kwargs={
                    "y": {
                        "text_embedding": text_embedding,
                        "history_motion_normalized": history,
                    }
                },
                device=device,
                ddim_steps=args.ddim_steps,
            )
            latent = latent.permute(1, 0, 2)
            future = vae.decode(latent, history, cfg["future_length"], scale_latent=True)
            generated.append(dataset.denormalize(future.squeeze(0)).detach().cpu())
            # Canonicalization note: The 272-dim HumanML3D representation is inherently
            # root-relative (joint positions relative to root, root velocity as separate
            # channels). No explicit coordinate transform is needed between primitives —
            # the representation itself acts as the canonical frame.
            all_frames = torch.cat([history, future], dim=1)
            history = all_frames[:, -cfg["history_length"] :]

    sequence = torch.cat(generated, dim=0).detach().numpy()
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / "rollout")
    stem = args.text_prompt.replace(" ", "_").replace(",", "__").replace("*", "x")
    np.save(output_dir / f"{stem}.npy", sequence)
    metadata = {
        "text_prompt": args.text_prompt,
        "texts": texts,
        "shape": list(sequence.shape),
        "checkpoint": str(checkpoint_path),
        "guidance_scale": args.guidance_scale,
        "ddim_steps": args.ddim_steps,
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"saved rollout to {output_dir / f'{stem}.npy'}")


if __name__ == "__main__":
    main()
