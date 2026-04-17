"""Deterministic toy backends for FloodForecaster model tests."""

import torch
import torch.nn as nn


class FakeGNOIn(nn.Module):
    def __init__(self, fno_hidden_channels: int, scale: float):
        super().__init__()
        self.fno_hidden_channels = fno_hidden_channels
        self.scale = scale

    def forward(self, y: torch.Tensor, x: torch.Tensor, f_y: torch.Tensor) -> torch.Tensor:
        del x
        mean_features = f_y.mean(dim=-1, keepdim=True)
        coord_term = y.unsqueeze(0)[..., :1] * self.scale
        offsets = torch.arange(
            self.fno_hidden_channels, dtype=f_y.dtype, device=f_y.device
        ).view(1, 1, -1)
        return mean_features + coord_term + offsets * self.scale


class FakeLatentEmbedding(nn.Module):
    def forward(self, in_p: torch.Tensor, ada_in=None) -> torch.Tensor:
        latent = in_p.permute(0, 3, 1, 2).contiguous()
        if ada_in is not None:
            latent = latent + 0.0 * ada_in.reshape(ada_in.shape[0], 1, 1, 1)
        return latent


class FakeGNOOut(nn.Module):
    def __init__(self, fno_hidden_channels: int, scale: float):
        super().__init__()
        self.fno_hidden_channels = fno_hidden_channels
        self.scale = scale

    def forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        f_y,
        f_x: torch.Tensor,
        reduction: str = "sum",
    ) -> torch.Tensor:
        del f_y, reduction
        query_count = y.shape[0]
        base = f_x.mean(dim=1, keepdim=True).expand(-1, query_count, -1)
        query_term = y.sum(dim=-1, keepdim=True).unsqueeze(0) * self.scale
        latent_term = x.mean().view(1, 1, 1) * self.scale
        return base + query_term + latent_term


class FakeProjection(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        weight = torch.arange(
            out_channels * hidden_channels, dtype=torch.float32
        ).reshape(out_channels, hidden_channels)
        weight = weight / max(hidden_channels, 1)
        bias = torch.linspace(-0.2, 0.2, out_channels, dtype=torch.float32)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhn,oh->bon", x, self.weight) + self.bias.view(1, -1, 1)


class FakeGINOBackbone(nn.Module):
    """Small deterministic GINO-like backend for wrapper tests."""

    def __init__(
        self,
        fno_hidden_channels: int = 4,
        out_channels: int = 3,
        scale: float = 0.25,
    ):
        super().__init__()
        self.fno_hidden_channels = fno_hidden_channels
        self.out_channels = out_channels
        self.scale = scale
        self.in_coord_dim_reverse_order = (2, 3)
        self.gno_in = FakeGNOIn(fno_hidden_channels, scale)
        self.latent_embedding = FakeLatentEmbedding()
        self.gno_out = FakeGNOOut(fno_hidden_channels, scale)
        self.projection = FakeProjection(fno_hidden_channels, out_channels)
