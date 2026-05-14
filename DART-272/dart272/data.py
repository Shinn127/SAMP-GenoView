from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TextSegment:
    caption: str
    start_t: float
    end_t: float

    @property
    def is_global(self) -> bool:
        return self.start_t == 0.0 and self.end_t == 0.0


def intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) >= max(a0, b0)


def load_text_segments(path: Path) -> list[TextSegment]:
    segments: list[TextSegment] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        caption, _, f_tag, to_tag = line.split("#")
        start_t = float(f_tag) if f_tag != "nan" else 0.0
        end_t = float(to_tag) if to_tag != "nan" else 0.0
        segments.append(TextSegment(caption=caption.strip(), start_t=start_t, end_t=end_t))
    return segments


class HumanML3D272Dataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        history_length: int,
        future_length: int,
        num_primitives: int,
        fps: int = 30,
        text_tolerance: float = 0.0,
        stride: int | None = None,
        clip_cache_path: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.motion_dir = self.data_root / "motion_data"
        self.text_dir = self.data_root / "texts"
        self.split = split
        self.history_length = history_length
        self.future_length = future_length
        self.num_primitives = num_primitives
        self.fps = fps
        self.text_tolerance = text_tolerance
        self.feature_dim = 272
        self.seq_length = history_length + future_length * num_primitives
        self.stride = future_length if stride is None else stride

        self.mean = torch.from_numpy(np.load(self.data_root / "mean_std" / "Mean.npy")).float()
        self.std = torch.from_numpy(np.load(self.data_root / "mean_std" / "Std.npy")).float().clamp(min=1e-6)

        # Load precomputed CLIP embeddings if available
        self.clip_cache = None
        self.text_to_idx: dict[str, int] = {}
        cache_path = Path(clip_cache_path) if clip_cache_path else self.data_root / ".cache" / "clip_embeddings.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu")
            self.clip_cache = cache["embeddings"]  # [num_texts, 512]
            self.text_to_idx = cache["text_to_idx"]

        split_ids = [line.strip() for line in (self.data_root / "split" / f"{split}.txt").read_text().splitlines() if line.strip()]
        self.motion_index: list[tuple[str, int]] = []
        self.lengths: dict[str, int] = {}
        self.text_segments: dict[str, list[TextSegment]] = {}
        self.global_captions: dict[str, list[str]] = {}
        self.all_texts: list[str] = []

        for sample_id in split_ids:
            motion_path = self.motion_dir / f"{sample_id}.npy"
            text_path = self.text_dir / f"{sample_id}.txt"
            if not motion_path.exists() or not text_path.exists():
                continue
            motion = np.load(motion_path, mmap_mode="r")
            if motion.ndim != 2 or motion.shape[1] != self.feature_dim:
                continue
            if motion.shape[0] < self.seq_length:
                continue
            self.lengths[sample_id] = int(motion.shape[0])
            segments = load_text_segments(text_path)
            self.text_segments[sample_id] = segments
            self.global_captions[sample_id] = [seg.caption for seg in segments if seg.is_global]
            self.all_texts.extend(seg.caption for seg in segments)
            max_start = motion.shape[0] - self.seq_length
            for start in range(0, max_start + 1, self.stride):
                self.motion_index.append((sample_id, start))

    def __len__(self) -> int:
        return len(self.motion_index)

    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean.to(motion.device)) / self.std.to(motion.device)

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std.to(motion.device) + self.mean.to(motion.device)

    def _select_texts(self, sample_id: str, start_frame: int) -> list[str]:
        segments = self.text_segments[sample_id]
        global_captions = self.global_captions.get(sample_id, [])
        texts: list[str] = []
        for primitive_idx in range(self.num_primitives):
            future_start = (start_frame + primitive_idx * self.future_length + self.history_length) / self.fps
            future_end = (start_frame + (primitive_idx + 1) * self.future_length + self.history_length - 1) / self.fps
            matched = [
                seg.caption
                for seg in segments
                if not seg.is_global
                and intervals_overlap(
                    seg.start_t,
                    seg.end_t,
                    future_start - self.text_tolerance,
                    future_end + self.text_tolerance,
                )
            ]
            if not matched and global_captions:
                matched = global_captions
            texts.append(random.choice(matched) if matched else "")
        return texts

    def __getitem__(self, index: int) -> dict:
        sample_id, start_frame = self.motion_index[index]
        motion = np.load(self.motion_dir / f"{sample_id}.npy")[start_frame : start_frame + self.seq_length]
        motion = torch.from_numpy(motion).float()
        texts = self._select_texts(sample_id, start_frame)

        item = {
            "sample_id": sample_id,
            "start_frame": start_frame,
            "motion": self.normalize(motion),
            "texts": texts,
        }

        # If CLIP cache is available, include precomputed embeddings
        if self.clip_cache is not None:
            indices = [self.text_to_idx.get(t, 0) for t in texts]
            item["text_embeddings"] = self.clip_cache[indices]  # [num_primitives, 512]

        return item


def collate_batch(batch: list[dict]) -> dict:
    result = {
        "sample_ids": [item["sample_id"] for item in batch],
        "start_frames": torch.tensor([item["start_frame"] for item in batch], dtype=torch.long),
        "motion": torch.stack([item["motion"] for item in batch], dim=0),
        "texts": [item["texts"] for item in batch],
    }
    # Include precomputed text embeddings if available
    if "text_embeddings" in batch[0]:
        result["text_embeddings"] = torch.stack([item["text_embeddings"] for item in batch], dim=0)  # [B, num_primitives, 512]
    return result
