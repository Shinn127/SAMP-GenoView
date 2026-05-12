from __future__ import annotations

import clip
import torch
import torch.nn as nn
from pathlib import Path


def get_clip_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".cache" / "clip"


def load_and_freeze_clip(clip_version: str = "ViT-B/32", device: str | torch.device = "cpu") -> nn.Module:
    cache_dir = get_clip_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, _ = clip.load(clip_version, device=device, jit=False, download_root=str(cache_dir))
    clip.model.convert_weights(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def encode_text(
    clip_model: nn.Module,
    texts: list[str],
    force_empty_zero: bool = True,
) -> torch.Tensor:
    device = next(clip_model.parameters()).device
    tokenized = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokenized).float()
    if force_empty_zero:
        empty_mask = torch.tensor([text.strip() == "" for text in texts], device=device, dtype=torch.bool)
        if empty_mask.any():
            embedding[empty_mask] = 0.0
    return embedding


if __name__ == "__main__":
    model = load_and_freeze_clip()
    texts = ["a photo of a cat", "a photo of a dog", ""]
    embeddings = encode_text(model, texts)
    print(embeddings)
