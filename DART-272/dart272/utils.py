from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    if resolved.type == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("MPS was requested but this PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this environment.")
    return resolved
