"""MLP actor-critic policy network for RL control.

Implements a PolicyNetwork that maps 1061D observations to 256D latent noise
actions for the diffusion denoiser, with a shared embedding encoder and
separate actor/critic heads.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions.normal import Normal

from control.config import PolicyArgs


class MLP(nn.Module):
    """Single hidden-layer MLP with configurable activation.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    h_dims : list[int]
        Hidden layer dimensions. Each entry adds one linear + activation layer.
    activation : str
        Activation function name: "tanh", "relu", "sigmoid", "gelu", or "lrelu".
    """

    def __init__(self, in_dim: int, h_dims: list[int], activation: str = "tanh") -> None:
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim_, h_dim))
            in_dim_ = h_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MLPBlock(nn.Module):
    """MLP block with residual connections and a final output projection.

    Each block consists of ``n_blocks`` residual MLP layers (each with two
    linear layers and activation), followed by a linear output projection.

    Parameters
    ----------
    h_dim : int
        Hidden dimension (input and internal dimension for residual layers).
    out_dim : int
        Output dimension of the final linear projection.
    n_blocks : int
        Number of residual MLP layers.
    actfun : str
        Activation function name for internal MLPs.
    residual : bool
        Whether to use residual (skip) connections.
    """

    def __init__(
        self,
        h_dim: int,
        out_dim: int,
        n_blocks: int,
        actfun: str = "relu",
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            [MLP(h_dim, h_dims=[h_dim, h_dim], activation=actfun) for _ in range(n_blocks)]
        )
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y


class PolicyNetwork(nn.Module):
    """MLP actor-critic policy for goal-reaching control.

    Architecture:
        Observation (1061D)
        ├── motion (544D) → MLP(544, 512) → motion_emb (512D)
        ├── text (512D) → MLP(512, 512) → text_emb (512D)
        ├── goal (4D) → MLP(4, 512) → goal_emb (512D)
        └── floor (1D) → MLP(1, 512) → floor_emb (512D)
                                            ↓
                            Concat (2048D) → MLP(2048, 512) → embedding (512D)
                                            ↓
                            ┌───────────────┴───────────────┐
                            Actor MLPBlock (2 res blocks)    Critic MLPBlock (2 res blocks)
                            → action_mean (256D)            → value (1D)
                            + learnable log_std (256D)

    Parameters
    ----------
    args : PolicyArgs
        Policy configuration dataclass.
    """

    # Observation layout constants
    OBS_DIM = 1061
    MOTION_DIM = 544
    TEXT_DIM = 512
    GOAL_DIM = 4  # goal_dir (3) + goal_dist (1)
    FLOOR_DIM = 1
    ACTION_DIM = 256

    # Observation index ranges
    GOAL_DIR_START = 0
    GOAL_DIR_END = 3
    GOAL_DIST_START = 3
    GOAL_DIST_END = 4
    TEXT_START = 4
    TEXT_END = 516
    MOTION_START = 516
    MOTION_END = 1060
    FLOOR_START = 1060
    FLOOR_END = 1061

    def __init__(self, args: PolicyArgs) -> None:
        super().__init__()
        self.args = args
        latent_dim = args.latent_dim
        activation = args.activation
        n_blocks = args.n_blocks

        # Individual modality encoders
        self.motion_encoder = MLP(
            in_dim=self.MOTION_DIM, h_dims=[latent_dim], activation=activation
        )
        self.text_encoder = MLP(
            in_dim=self.TEXT_DIM, h_dims=[latent_dim], activation=activation
        )
        self.goal_encoder = MLP(
            in_dim=self.GOAL_DIM, h_dims=[latent_dim], activation=activation
        )
        self.floor_encoder = MLP(
            in_dim=self.FLOOR_DIM, h_dims=[latent_dim], activation=activation
        )

        # Embedding encoder: fuses all modality embeddings
        self.embedding_encoder = MLP(
            in_dim=latent_dim * 4, h_dims=[latent_dim], activation=activation
        )

        # Actor head: outputs action mean (256D)
        self.actor = MLPBlock(
            latent_dim, self.ACTION_DIM, n_blocks, actfun=activation
        )

        # Learnable log-standard-deviation parameter
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.ACTION_DIM))

        # Critic head: outputs scalar value
        self.critic = MLPBlock(latent_dim, 1, n_blocks, actfun=activation)

        # Optional zero initialization for actor
        if args.use_zero_init:
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

    def get_embedding(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation components and fuse into a single embedding.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (B, 1061).

        Returns
        -------
        torch.Tensor
            Fused embedding of shape (B, latent_dim).
        """
        # Extract observation components
        goal = obs[:, self.GOAL_DIR_START : self.GOAL_DIST_END]  # (B, 4)
        text = obs[:, self.TEXT_START : self.TEXT_END]  # (B, 512)
        motion = obs[:, self.MOTION_START : self.MOTION_END]  # (B, 544)
        floor = obs[:, self.FLOOR_START : self.FLOOR_END]  # (B, 1)

        # Encode each modality
        motion_emb = self.motion_encoder(motion)
        text_emb = self.text_encoder(text)
        goal_emb = self.goal_encoder(goal)
        floor_emb = self.floor_encoder(floor)

        # Concatenate and fuse
        concat = torch.cat((motion_emb, text_emb, goal_emb, floor_emb), dim=1)
        embedding = self.embedding_encoder(concat)
        return embedding

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action, log probability, entropy, and value from observation.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (B, 1061).
        action : torch.Tensor | None
            If provided, compute log_prob for this action instead of sampling.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (action, log_prob, entropy, value) where:
            - action: (B, 256)
            - log_prob: (B,)
            - entropy: (B,)
            - value: (B,)
        """
        embedding = self.get_embedding(obs)

        # Actor: compute action mean
        action_mean = self.actor(embedding)
        if self.args.use_tanh_scale:
            action_mean = torch.tanh(action_mean) * 4.0

        # Clamp log_std and compute std
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_logstd = action_logstd.clamp(
            min=self.args.min_log_std, max=self.args.max_log_std
        )
        action_std = torch.exp(action_logstd)

        # Normal distribution
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)

        # Critic: compute value
        value = self.critic(embedding).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from observation.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (B, 1061).

        Returns
        -------
        torch.Tensor
            Value estimate of shape (B,).
        """
        embedding = self.get_embedding(obs)
        return self.critic(embedding).squeeze(-1)
