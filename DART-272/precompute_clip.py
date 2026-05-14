"""Precompute CLIP text embeddings for all unique captions in the dataset.

Saves a cache file containing:
  - vocab: list of unique text strings (index 0 = empty string)
  - embeddings: [num_texts, 512] float32 tensor

Usage:
    python DART-272/precompute_clip.py --data-root humanml3d_272 --output humanml3d_272/.cache/clip_embeddings.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dart272.data import HumanML3D272Dataset
from dart272.text import encode_text, load_and_freeze_clip
from dart272.utils import ensure_dir, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="humanml3d_272")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Defaults to <data-root>/.cache/clip_embeddings.pt")
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for CLIP encoding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_root = Path(args.data_root)
    output_path = Path(args.output) if args.output else data_root / ".cache" / "clip_embeddings.pt"
    ensure_dir(output_path.parent)

    print(f"Using device: {device}")
    print(f"Loading dataset from: {data_root}")

    # Collect all unique texts from all splits
    all_texts = set()
    for split in ["train", "val", "test"]:
        split_file = data_root / "split" / f"{split}.txt"
        if not split_file.exists():
            continue
        # Use a dummy dataset just to collect texts
        ds = HumanML3D272Dataset(
            data_root=str(data_root),
            split=split,
            history_length=2,
            future_length=8,
            num_primitives=4,
        )
        for segments in ds.text_segments.values():
            for seg in segments:
                all_texts.add(seg.caption)

    # Always include empty string at index 0
    all_texts.discard("")
    vocab = [""] + sorted(all_texts)
    print(f"  Unique captions: {len(vocab)} (including empty)")

    # Encode in batches
    clip_model = load_and_freeze_clip(args.clip_version, device=device)
    embeddings = []
    for i in range(0, len(vocab), args.batch_size):
        batch_texts = vocab[i : i + args.batch_size]
        emb = encode_text(clip_model, batch_texts, force_empty_zero=True)
        embeddings.append(emb.cpu())
        if (i // args.batch_size) % 10 == 0:
            print(f"  Encoded {min(i + args.batch_size, len(vocab))}/{len(vocab)}")

    embeddings = torch.cat(embeddings, dim=0)  # [num_texts, 512]
    assert embeddings.shape[0] == len(vocab)

    # Build text-to-index lookup
    text_to_idx = {text: idx for idx, text in enumerate(vocab)}

    payload = {
        "vocab": vocab,
        "embeddings": embeddings,
        "text_to_idx": text_to_idx,
        "clip_version": args.clip_version,
    }
    torch.save(payload, output_path)
    print(f"  Saved {len(vocab)} embeddings to {output_path}")
    print(f"  Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
