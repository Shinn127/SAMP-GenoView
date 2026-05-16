"""Async inference engine for real-time text-to-motion generation in GenoView.

This module wraps the DART-272 pipeline (CLIP + Denoiser + VAE) and runs inference
in a background thread, feeding generated 272-dim motion frames into a queue that
the render loop consumes at 30fps.

Usage:
    engine = InferenceEngine(mld_checkpoint_path, data_root, device="mps")
    engine.start()
    engine.set_text("a person walks forward")
    # In render loop:
    frame = engine.consume_frame()  # returns [272] ndarray or None
    engine.stop()
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Add DART-272 to path so we can import its modules
_DART_DIR = Path(__file__).resolve().parent.parent / "DART-272"
if str(_DART_DIR) not in sys.path:
    sys.path.insert(0, str(_DART_DIR))

from dart272.data import HumanML3D272Dataset
from dart272.denoiser import ClassifierFreeGuidanceWrapper, DenoiserMLP, DenoiserTransformer
from dart272.diffusion import GaussianDiffusion
from dart272.text import encode_text, load_and_freeze_clip
from dart272.utils import load_json
from dart272.vae import AutoMldVae


def _build_denoiser(cfg: dict, cond_mask_prob: float = 0.1) -> torch.nn.Module:
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


class InferenceEngine:
    """Background inference engine for text-conditioned motion generation.

    Architecture:
        - Main thread: sets text prompts, consumes generated frames
        - Worker thread: runs CLIP + diffusion + VAE decode loop

    The engine maintains a frame buffer (deque). When the buffer runs low,
    the worker generates the next primitive (8 frames) and appends them.

    Continuous Control:
        - set_text(): Seamlessly switch to a new action without resetting history.
          The next generated primitive will use the new text while maintaining
          motion continuity from the autoregressive history.
        - queue_text(): Schedule a sequence of actions. Each text is used for one
          primitive, then the next text in the queue is consumed.
        - interrupt(): Immediately discard buffered frames and start generating
          with the current text from the latest history state.
    """

    def __init__(
        self,
        mld_checkpoint: str | Path,
        data_root: str | Path,
        device: str = "auto",
        guidance_scale: float = 5.0,
        ddim_steps: int = 5,
        buffer_size: int = 16,
        seed_motion_path: str | Path | None = None,
        seed_start_frame: int = 0,
    ) -> None:
        self.device = self._resolve_device(device)
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.buffer_size = buffer_size

        # Load models
        checkpoint_path = Path(mld_checkpoint)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.cfg = load_json(checkpoint_path.parent / "config.json")

        # Resolve VAE checkpoint path (may be relative to training working dir)
        mvae_ckpt_path = Path(ckpt["vae_checkpoint"])
        if not mvae_ckpt_path.exists():
            # Try sibling directory with same folder name under outputs/
            # e.g. stored "checkpoints/mvae_run2/checkpoint_best.pt"
            #   → look in outputs/mvae_run2/checkpoint_best.pt
            vae_dir_name = mvae_ckpt_path.parent.name
            vae_file_name = mvae_ckpt_path.name
            candidate = checkpoint_path.parent.parent / vae_dir_name / vae_file_name
            if candidate.exists():
                mvae_ckpt_path = candidate
        if not mvae_ckpt_path.exists():
            # Try relative to the MLD checkpoint's grandparent
            mvae_ckpt_path = checkpoint_path.parent.parent / mvae_ckpt_path.name
        if not mvae_ckpt_path.exists():
            # Try relative to DART-272 root (one level above outputs/)
            dart_root = checkpoint_path.parent.parent.parent
            mvae_ckpt_path = dart_root / ckpt["vae_checkpoint"]
        if not mvae_ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot find VAE checkpoint. Stored path: '{ckpt['vae_checkpoint']}'. "
                f"Searched near: {checkpoint_path.parent.parent}. "
                f"Pass --mvae-ckpt to specify explicitly."
            )
        mvae_cfg = load_json(mvae_ckpt_path.parent / "config.json")
        mvae_ckpt = torch.load(mvae_ckpt_path, map_location="cpu")

        self.history_length = self.cfg["history_length"]
        self.future_length = self.cfg["future_length"]
        self.feature_dim = self.cfg["feature_dim"]

        # Dataset for normalize/denormalize
        self.data_root = Path(data_root)
        if not self.data_root.is_absolute():
            self.data_root = self.data_root.resolve()
        self.mean = torch.from_numpy(
            np.load(self.data_root / "mean_std" / "Mean.npy")
        ).float().to(self.device)
        self.std = torch.from_numpy(
            np.load(self.data_root / "mean_std" / "Std.npy")
        ).float().clamp(min=1e-6).to(self.device)

        # VAE
        self.vae = AutoMldVae(
            nfeats=mvae_cfg["feature_dim"],
            latent_dim=(mvae_cfg["latent_size"], mvae_cfg["latent_width"]),
            h_dim=mvae_cfg["h_dim"],
            ff_size=mvae_cfg["ff_size"],
            num_layers=mvae_cfg["num_layers"],
            num_heads=mvae_cfg["num_heads"],
            dropout=mvae_cfg["dropout"],
        ).to(self.device)
        self.vae.load_state_dict(mvae_ckpt["model_state"])
        self.vae.eval()

        # CLIP
        clip_version = self.cfg.get("clip_version", "ViT-B/32")
        self.clip_model = load_and_freeze_clip(clip_version, device=self.device)

        # Denoiser
        denoiser = _build_denoiser(self.cfg, cond_mask_prob=0.1).to(self.device)
        denoiser.load_state_dict(ckpt["denoiser_state"])
        denoiser.eval()
        if guidance_scale > 1.0:
            self.model = ClassifierFreeGuidanceWrapper(denoiser, guidance_scale=guidance_scale)
        else:
            self.model = denoiser

        # Diffusion
        self.diffusion = GaussianDiffusion(num_steps=self.cfg["diffusion_steps"])

        # Initialize history from seed motion or zeros
        if seed_motion_path is not None:
            seed = torch.from_numpy(np.load(seed_motion_path)).float()
            start = seed_start_frame
            history_raw = seed[start : start + self.history_length]
            self._history = self._normalize(history_raw.to(self.device)).unsqueeze(0)
        else:
            # Use a T-pose / zero motion as initial history
            self._history = torch.zeros(
                1, self.history_length, self.feature_dim, device=self.device
            )

        # Thread state
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._current_text: str = ""
        self._text_lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._buffer_low_threshold = self.future_length  # refill when < 8 frames

        # Continuous control state
        self._text_queue: deque[str] = deque()  # scheduled text sequence
        self._continuous_mode: bool = True  # keep generating with current text
        self._interrupt_flag: bool = False  # signal to discard buffer and restart
        self._primitives_generated: int = 0  # count for status display

        # Cached text embedding (avoid re-encoding same text)
        self._cached_text: str = ""
        self._cached_text_embedding: torch.Tensor | None = None

        print(f"[InferenceEngine] Loaded on {self.device}")
        print(f"  history_length={self.history_length}, future_length={self.future_length}")
        print(f"  guidance_scale={guidance_scale}, ddim_steps={ddim_steps}")
        print(f"  continuous_mode=True")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _normalize(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean) / self.std

    def _denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std + self.mean

    def set_text(self, text: str, interrupt: bool = False) -> None:
        """Set the current text prompt for continuous generation. Thread-safe.

        In continuous mode, the next primitive will seamlessly use this new text
        while maintaining motion continuity (no history reset).

        Args:
            text: The new text prompt.
            interrupt: If True, discard buffered frames and start generating
                       immediately with the new text from current history.
        """
        with self._text_lock:
            self._current_text = text.strip()
            if interrupt:
                self._interrupt_flag = True

    def get_text(self) -> str:
        """Get the current text prompt. Thread-safe."""
        with self._text_lock:
            return self._current_text

    def queue_text(self, text: str, repeat: int = 1) -> None:
        """Add text to the generation queue. Each entry generates one primitive.

        After the queue is exhausted, falls back to the current text (set via set_text).

        Args:
            text: Text prompt to queue.
            repeat: Number of primitives to generate with this text.
        """
        with self._text_lock:
            for _ in range(repeat):
                self._text_queue.append(text.strip())

    def clear_queue(self) -> None:
        """Clear the text queue."""
        with self._text_lock:
            self._text_queue.clear()

    def queue_length(self) -> int:
        """Number of texts remaining in the queue."""
        return len(self._text_queue)

    def interrupt(self) -> None:
        """Interrupt current generation: discard buffered frames and regenerate.

        The next primitive will start from the most recent history state,
        using the current text. This provides immediate responsiveness when
        the user changes direction.
        """
        with self._text_lock:
            self._interrupt_flag = True

    @property
    def continuous_mode(self) -> bool:
        return self._continuous_mode

    @continuous_mode.setter
    def continuous_mode(self, value: bool) -> None:
        self._continuous_mode = value

    @property
    def primitives_generated(self) -> int:
        return self._primitives_generated

    def start(self) -> None:
        """Start the background inference worker."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print("[InferenceEngine] Worker started.")

    def stop(self) -> None:
        """Stop the background inference worker."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        print("[InferenceEngine] Worker stopped.")

    def consume_frame(self) -> np.ndarray | None:
        """Pop the next 272-dim frame from the buffer. Returns None if empty."""
        if self._frame_buffer:
            return self._frame_buffer.popleft()
        return None

    def frames_available(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._frame_buffer)

    def is_running(self) -> bool:
        return self._running

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get CLIP embedding, using cache if text hasn't changed."""
        if text == self._cached_text and self._cached_text_embedding is not None:
            return self._cached_text_embedding
        with torch.no_grad():
            embedding = encode_text(self.clip_model, [text], force_empty_zero=True)
        self._cached_text = text
        self._cached_text_embedding = embedding
        return embedding

    @torch.no_grad()
    def _generate_primitive(self, text: str) -> np.ndarray:
        """Generate one primitive (future_length frames) of 272-dim motion.

        Returns denormalized motion as numpy array [future_length, 272].
        """
        text_embedding = self._get_text_embedding(text)

        # Diffusion sampling
        shape = (1, self.cfg["latent_size"], self.cfg["latent_width"])
        model_kwargs = {
            "y": {
                "text_embedding": text_embedding,
                "history_motion_normalized": self._history,
            }
        }

        if self.ddim_steps > 0:
            latent = self.diffusion.ddim_sample_loop(
                self.model, shape=shape, model_kwargs=model_kwargs,
                device=self.device, ddim_steps=self.ddim_steps,
            )
        else:
            latent = self.diffusion.p_sample_loop(
                self.model, shape=shape, model_kwargs=model_kwargs,
                device=self.device,
            )

        latent = latent.permute(1, 0, 2)  # [latent_size, B, width] -> for VAE

        # VAE decode
        future_normalized = self.vae.decode(
            latent, self._history, self.future_length, scale_latent=True
        )

        # Update history for next primitive
        all_frames = torch.cat([self._history, future_normalized], dim=1)
        self._history = all_frames[:, -self.history_length:]

        # Denormalize and return
        future_denorm = self._denormalize(future_normalized.squeeze(0))
        return future_denorm.cpu().numpy().astype(np.float32)

    def _worker_loop(self) -> None:
        """Background worker that keeps the frame buffer filled.

        Continuous control logic:
        1. Check for interrupt flag → clear buffer, continue with current text
        2. Pop from text queue if available, otherwise use current text
        3. Generate primitive and append frames to buffer
        4. In continuous mode, keep generating even when buffer is healthy
        """
        while self._running:
            # Handle interrupt: discard buffered frames immediately
            with self._text_lock:
                if self._interrupt_flag:
                    self._frame_buffer.clear()
                    self._interrupt_flag = False
                    print("[InferenceEngine] Interrupted — buffer cleared.")

            # Determine which text to use
            with self._text_lock:
                if self._text_queue:
                    text = self._text_queue.popleft()
                else:
                    text = self._current_text

            # If no text set, sleep briefly
            if not text:
                time.sleep(0.01)
                continue

            # In continuous mode: always generate if buffer isn't completely full
            # In non-continuous mode: only generate when buffer is low
            buffer_len = len(self._frame_buffer)
            should_generate = False
            if self._continuous_mode:
                should_generate = buffer_len < self.buffer_size - self.future_length
            else:
                should_generate = buffer_len < self._buffer_low_threshold

            if should_generate:
                try:
                    frames = self._generate_primitive(text)
                    # Check interrupt again before appending (in case it fired during generation)
                    with self._text_lock:
                        if self._interrupt_flag:
                            self._frame_buffer.clear()
                            self._interrupt_flag = False
                            print("[InferenceEngine] Interrupted during generation — discarded.")
                            continue
                    for i in range(len(frames)):
                        self._frame_buffer.append(frames[i])
                    self._primitives_generated += 1
                except Exception as e:
                    print(f"[InferenceEngine] Error in generation: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.005)

    def reset_history(self, seed_motion: np.ndarray | None = None, start_frame: int = 0) -> None:
        """Reset the autoregressive history. Clears the frame buffer and queue."""
        self._frame_buffer.clear()
        with self._text_lock:
            self._text_queue.clear()
            self._interrupt_flag = False
        self._primitives_generated = 0
        if seed_motion is not None:
            seed = torch.from_numpy(seed_motion).float()
            history_raw = seed[start_frame : start_frame + self.history_length]
            self._history = self._normalize(history_raw.to(self.device)).unsqueeze(0)
        else:
            self._history = torch.zeros(
                1, self.history_length, self.feature_dim, device=self.device
            )

    def get_status(self) -> dict:
        """Get engine status for UI display."""
        return {
            "text": self.get_text(),
            "queue_length": len(self._text_queue),
            "buffer_frames": len(self._frame_buffer),
            "primitives_generated": self._primitives_generated,
            "continuous_mode": self._continuous_mode,
            "running": self._running,
        }
