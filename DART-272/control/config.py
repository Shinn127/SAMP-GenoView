"""Configuration dataclasses for the RL control policy module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvArgs:
    """Environment configuration for the vectorized RL environment.

    Attributes:
        checkpoint_path: Path to the MLD checkpoint file.
        seed_data_path: Path to directory containing seed .npy files.
        data_root: Path to dataset root (for loading mean/std arrays).
        num_envs: Number of parallel environments for batched rollouts.
        max_steps: Maximum steps per episode before truncation.
        texts: List of locomotion text prompts for CLIP conditioning.
        guidance_scale: Classifier-free guidance scale for denoiser.
        ddim_steps: Number of DDIM sampling steps per motion primitive.
        success_threshold: XY distance (meters) below which goal is reached.
        terminate_threshold: XY distance (meters) above which episode terminates.
        obs_goal_angle_clip: Maximum goal direction angle in observation (degrees).
        obs_goal_dist_clip: Maximum goal distance value in observation (meters).
    """

    checkpoint_path: str
    seed_data_path: str
    data_root: str
    num_envs: int = 256
    max_steps: int = 256
    texts: list[str] = field(default_factory=lambda: ["walk", "run", "hop on left leg"])
    guidance_scale: float = 5.0
    ddim_steps: int = 10
    success_threshold: float = 0.3
    terminate_threshold: float = 100.0
    obs_goal_angle_clip: float = 60.0
    obs_goal_dist_clip: float = 5.0
    enable_export: bool = False
    export_interval: int = 100
    max_export: int = 16
    export_dir: str = ""


@dataclass
class PolicyArgs:
    """Policy network architecture configuration.

    Attributes:
        latent_dim: Shared latent dimension for all encoders and embedding.
        n_blocks: Number of residual blocks in actor and critic MLPBlocks.
        activation: Activation function name (e.g. "tanh", "relu").
        min_log_std: Minimum value for clamped log-standard-deviation.
        max_log_std: Maximum value for clamped log-standard-deviation.
        use_tanh_scale: If True, apply tanh scaled by 4.0 to action mean.
        use_zero_init: If True, initialize actor MLPBlock weights with 0.01 scale.
    """

    latent_dim: int = 512
    n_blocks: int = 2
    activation: str = "tanh"
    min_log_std: float = -1.0
    max_log_std: float = 1.0
    use_tanh_scale: bool = False
    use_zero_init: bool = False


@dataclass
class GoalSchedulerArgs:
    """Goal scheduler curriculum configuration.

    Attributes:
        dist_min: Minimum goal distance (meters).
        dist_max_init: Initial maximum goal distance (meters).
        dist_max_delta: Distance increment per curriculum step (meters).
        dist_max_clamp: Maximum allowed goal distance (meters).
        angle_init: Initial goal angle range (degrees).
        angle_delta: Angle range increment per curriculum step (degrees).
        angle_max: Maximum goal angle range (degrees).
        curriculum_interval: Global timesteps between curriculum steps.
    """

    dist_min: float = 0.5
    dist_max_init: float = 2.0
    dist_max_delta: float = 1.0
    dist_max_clamp: float = 5.0
    angle_init: float = 0.0
    angle_delta: float = 120.0
    angle_max: float = 360.0
    curriculum_interval: int = 50000


@dataclass
class RewardWeights:
    """Weights for multi-component reward function.

    Attributes:
        weight_success: Weight for goal-reaching success reward.
        weight_dist: Weight for distance reduction reward.
        weight_foot_floor: Weight for foot-floor contact penalty.
        weight_skate: Weight for foot skating penalty.
        weight_orient: Weight for orientation alignment reward.
    """

    weight_success: float = 10.0
    weight_dist: float = 1.0
    weight_foot_floor: float = 1.0
    weight_skate: float = 1.0
    weight_orient: float = 1.0


@dataclass
class TrainArgs:
    """PPO training loop configuration.

    Attributes:
        env_args: Environment configuration.
        policy_args: Policy network configuration.
        goal_args: Goal scheduler configuration.
        reward_weights: Reward function weights.
        learning_rate: Initial learning rate for Adam optimizer.
        gamma: Discount factor for returns.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        clip_coef: PPO clipping coefficient for policy ratio.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        update_epochs: Number of PPO update epochs per rollout.
        minibatch_size: Minibatch size for PPO updates.
        num_steps: Rollout length (steps per environment per iteration).
        num_iterations: Total number of training iterations.
        anneal_lr: Whether to linearly anneal learning rate to zero.
        save_interval: Iterations between checkpoint saves.
        export_interval: Iterations between rollout exports.
        max_export: Maximum rollouts to export per interval.
        save_dir: Directory for saving checkpoints and exports.
        use_wandb: Whether to log to Weights and Biases.
        wandb_project: W&B project name.
        seed: Random seed for reproducibility.
    """

    env_args: EnvArgs
    policy_args: PolicyArgs
    goal_args: GoalSchedulerArgs
    reward_weights: RewardWeights
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 1024
    num_steps: int = 32
    num_iterations: int = 500
    anneal_lr: bool = True
    # Logging
    save_interval: int = 100
    export_interval: int = 100
    max_export: int = 16
    save_dir: str = "DART-272/outputs/rl_control"
    use_wandb: bool = False
    wandb_project: str = "dart272-control"
    seed: int = 1
