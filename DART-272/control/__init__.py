"""RL Control Policy module for DART-272.

This module implements a PPO-based policy network that steers a character toward
goal locations by generating 256D latent noise vectors for the diffusion denoiser.
The system uses the 272-dim HumanML3D motion representation with direct joint
extraction via world_transform, eliminating the need for a body model.

Components:
    - env: Vectorized RL environment wrapping the DART-272 motion pipeline
    - policy: MLP actor-critic policy network
    - train: PPO training loop
    - test: Inference and evaluation script
    - goal_scheduler: Curriculum learning for progressive goal difficulty
    - reward: Multi-component reward function
    - config: Dataclass configurations for all components
"""
