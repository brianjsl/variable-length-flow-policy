import math
import torch
import torch.nn as nn
import numpy as np

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B, T]
        returns: [B, T, D]
        """
        if t.ndim not in [1, 2]:
            raise ValueError(f"Input t must be 1D or 2D, got {t.ndim} dimensions")

        device = t.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype = torch.float32) * -emb)
        emb = t[..., None] * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class FourierEmbedding(torch.nn.Module):
    """
    Adapted from EDM2 - https://github.com/NVlabs/edm2/blob/38d5a70fe338edc8b3aac4da8a0cefbc4a057fb8/training/networks_edm2.py#L73
    """

    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y[..., None] * self.freqs.to(torch.float32)
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)