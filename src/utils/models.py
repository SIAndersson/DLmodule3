import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP architecture for predicting both vector fields and noise.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        model_type: str = "vector_field",
    ):
        super().__init__()
        self.dim = input_dim
        self.time_embed_dim = time_embed_dim

        if model_type == "vector_field":
            activation = nn.SiLU
        elif model_type == "noise_predictor":
            activation = nn.ReLU
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Sinusoidal time embedding (similar to diffusion models)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network layers
        layers = [nn.Linear(self.dim + hidden_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.extend([activation(), nn.Linear(hidden_dim, hidden_dim)])
        layers.extend([activation(), nn.Linear(hidden_dim, self.dim)])

        self.net = nn.Sequential(*layers)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional embeddings for time.

        Uses sinusoidal functions sin(t/10000^(2i/d)) and cos(t/10000^(2i/d))
        to encode time information in a way that's smooth and periodic.
        """
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vector field network.

        Args:
            x: Input tensor of shape (batch_size, dim)
            t: Time tensor of shape (batch_size,)

        Returns:
            Vector field v_θ(x, t) of shape (batch_size, dim)
        """
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Concatenate spatial and temporal features
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


class CNN(nn.Module):
    """
    Simple CNN architecture for predicting both vector fields and noise, adapted for image input.
    """

    def __init__(
        self,
        input_channels: int = 3,
        time_embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        model_type: str = "vector_field",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim

        if model_type == "vector_field":
            activation = nn.SiLU
        elif model_type == "noise_predictor":
            activation = nn.ReLU
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )

        # This projects the time embedding to a spatial feature map
        # NOTE: Projecting to a single channel so that there is only one
        # extra dimension, 3 spatial dimensions and 1 time dimension
        self.time_proj = nn.Linear(hidden_dim, 1)

        # CNN backbone
        self.total_num_layers = 5 * num_layers
        convs = [nn.Conv2d(input_channels + 1, hidden_dim, kernel_size=3, padding=1)]
        for _ in range(num_layers - 1):
            convs.extend(
                [
                    activation(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                ]
            )
        convs.extend([activation(), nn.Conv2d(hidden_dim, input_channels, kernel_size=3, padding=1)])

        self.conv_net = nn.Sequential(*convs)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional embeddings for time.

        Uses sinusoidal functions sin(t/10000^(2i/d)) and cos(t/10000^(2i/d))
        to encode time information in a way that's smooth and periodic.
        """
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Time tensor of shape (B,)
        Returns:
            Tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)  # (B, 1)
        t_proj = self.time_proj(t_emb).view(B, 1, 1, 1)  # (B, 1, 1, 1)
        t_feat = t_proj.expand(-1, 1, H, W)  # (B, 1, H, W)

        # Concatenate time embedding as an additional channel
        x_cat = torch.cat([x, t_feat], dim=1)  # (B, C+1, H, W)
        return self.conv_net(x_cat)
