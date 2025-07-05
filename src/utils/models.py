import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        convs.extend(
            [
                activation(),
                nn.Conv2d(hidden_dim, input_channels, kernel_size=3, padding=1),
            ]
        )

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


# Time embedding model for the UNet model, which is an attempt to improve the face results
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# ResBlock for the UNet model
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb):
        h = self.block1(x)
        # Add time embedding
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)
    

# Attention block for the UNet model, to hopefully identify relevant features
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, eps=1e-5):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=channels, eps=eps)
        
        # Use PyTorch attention
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0  # Dropout handled in UNet
        )
        
    def forward(self, x):
        """
            x: (B, C, H, W)
            returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        h = h.view(B, C, H * W).transpose(1, 2)
        
        # Self-attention
        attn_out, _ = self.attention(h, h, h)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + attn_out
    

# UNet model, uses SiLU for everything and attention blocks to identify relevant features
# My earlier split between ReLU and SiLU was inspired by the original DDPM paper, but this is now outdated hence this uses SiLU for everything
# Hopefully skip connections in UNet and the attention will help to get the facial geometry correct instead of a soup of facial features
class UNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        time_emb_dim=128,
        base_channels=128,
        channel_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.time_emb_dim = time_emb_dim
        self.base_channels = base_channels
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        channels = [base_channels] + [base_channels * m for m in channel_mult]
        
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            # Resolution at this level
            resolution = 64 // (2 ** i)
            
            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, time_emb_dim * 4, dropout))
                in_ch = out_ch
            self.encoder_blocks.append(blocks)
            
            # Attention
            if resolution in attention_resolutions:
                self.encoder_attentions.append(AttentionBlock(out_ch))
            else:
                self.encoder_attentions.append(nn.Identity())
            
            # Downsample
            if i < len(channel_mult) - 1:
                self.encoder_downsample.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.encoder_downsample.append(nn.Identity())
        
        # Middle
        mid_channels = channels[-1]
        self.middle_block1 = ResBlock(mid_channels, mid_channels, time_emb_dim * 4, dropout)
        self.middle_attention = AttentionBlock(mid_channels)
        self.middle_block2 = ResBlock(mid_channels, mid_channels, time_emb_dim * 4, dropout)
        
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i, (in_ch, out_ch) in enumerate(zip(reversed(channels[1:]), reversed(channels[:-1]))):
            # Resolution at this level
            resolution = 64 // (2 ** (len(channel_mult) - 1 - i))
            
            # Residual blocks (note: input has skip connection, so double channels)
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):  # +1 for skip connection
                skip_ch = in_ch if j == 0 else 0
                blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim * 4, dropout))
                in_ch = out_ch
            self.decoder_blocks.append(blocks)
            
            # Attention
            if resolution in attention_resolutions:
                self.decoder_attentions.append(AttentionBlock(out_ch))
            else:
                self.decoder_attentions.append(nn.Identity())
            
            # Upsample
            if i < len(channel_mult) - 1:
                self.decoder_upsample.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.decoder_upsample.append(nn.Identity())
        
        # Output
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, input_channels, 3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder
        skip_connections = []
        for blocks, attention, downsample in zip(self.encoder_blocks, self.encoder_attentions, self.encoder_downsample):
            for block in blocks:
                h = block(h, t_emb)
            h = attention(h)
            skip_connections.append(h)
            h = downsample(h)
        
        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attention(h)
        h = self.middle_block2(h, t_emb)
        
        # Decoder
        for blocks, attention, upsample in zip(self.decoder_blocks, self.decoder_attentions, self.decoder_upsample):
            # First block uses skip connection
            skip = skip_connections.pop()
            h = blocks[0](torch.cat([h, skip], dim=1), t_emb)
            
            # Remaining blocks
            for block in blocks[1:]:
                h = block(h, t_emb)
            
            h = attention(h)
            h = upsample(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h