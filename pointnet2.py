"""A compact, pure-PyTorch PointNet++ classification backbone.

Tensor conventions:
    xyz:      (batch, 3, points)
    features: (batch, channels, points), or None

The neighbourhood operations are intentionally written in PyTorch so the
architecture is easy to inspect and change.  For large training runs they can
later be replaced by optimized CUDA operators without changing the modules.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def square_distance(source: Tensor, target: Tensor) -> Tensor:
    """Return pairwise squared distances for (B, N, C) and (B, M, C)."""
    return (
        (source**2).sum(dim=-1, keepdim=True)
        + (target**2).sum(dim=-1).unsqueeze(1)
        - 2 * source @ target.transpose(1, 2)
    ).clamp_min_(0)


def index_points(points: Tensor, indices: Tensor) -> Tensor:
    """Gather (B, N, C) points using indices shaped (B, ...)."""
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(points.shape[0], *([1] * (indices.ndim - 1)))
    return points[batch, indices]


def farthest_point_sample(xyz: Tensor, number: int) -> Tensor:
    """Choose point indices with iterative farthest-point sampling.

    The first centroid is deterministic (index 0), which makes tests and
    evaluation reproducible. ``number`` may not exceed the input point count.
    """
    batch_size, point_count, _ = xyz.shape
    if not 0 < number <= point_count:
        raise ValueError("number must be between 1 and the number of input points")

    centroids = torch.empty(batch_size, number, dtype=torch.long, device=xyz.device)
    distances = torch.full(
        (batch_size, point_count), torch.inf, dtype=xyz.dtype, device=xyz.device
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
    batch = torch.arange(batch_size, device=xyz.device)

    for position in range(number):
        centroids[:, position] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distances = torch.minimum(distances, ((xyz - centroid) ** 2).sum(dim=-1))
        farthest = distances.max(dim=-1).indices
    return centroids


def query_ball_point(radius: float, samples: int, xyz: Tensor, centroids: Tensor) -> Tensor:
    """Return up to ``samples`` neighbours inside each centroid's radius.

    Sparse neighbourhoods are padded with their nearest valid point, so every
    centroid always yields a dense, fixed-size tensor.
    """
    if radius <= 0 or samples <= 0:
        raise ValueError("radius and samples must be positive")

    distances = square_distance(centroids, xyz)
    point_count = xyz.shape[1]
    take = min(samples, point_count)
    masked = distances.masked_fill(distances > radius**2, torch.inf)
    selected_distances, indices = masked.topk(take, dim=-1, largest=False, sorted=False)

    nearest = distances.argmin(dim=-1, keepdim=True)
    indices = torch.where(torch.isinf(selected_distances), nearest.expand_as(indices), indices)
    if take < samples:
        padding = nearest.expand(*nearest.shape[:-1], samples - take)
        indices = torch.cat((indices, padding), dim=-1)
    return indices


def sample_and_group(
    number: int,
    radius: float,
    samples: int,
    xyz: Tensor,
    features: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Sample centroids and construct their normalized local neighbourhoods."""
    centroid_indices = farthest_point_sample(xyz, number)
    centroids = index_points(xyz, centroid_indices)
    neighbour_indices = query_ball_point(radius, samples, xyz, centroids)
    grouped_xyz = index_points(xyz, neighbour_indices) - centroids.unsqueeze(2)
    if features is None:
        grouped = grouped_xyz
    else:
        grouped = torch.cat((grouped_xyz, index_points(features, neighbour_indices)), dim=-1)
    return centroids, grouped


class PointNetSetAbstraction(nn.Module):
    """One PointNet++ set-abstraction (sampling, grouping, PointNet) layer."""

    def __init__(
        self,
        number: int | None,
        radius: float | None,
        samples: int | None,
        feature_channels: int,
        mlp_channels: Sequence[int],
        *,
        group_all: bool = False,
    ) -> None:
        super().__init__()
        if not mlp_channels:
            raise ValueError("mlp_channels cannot be empty")
        if not group_all and (number is None or radius is None or samples is None):
            raise ValueError("number, radius, and samples are required unless group_all=True")

        self.number = number
        self.radius = radius
        self.samples = samples
        self.group_all = group_all

        channels = [feature_channels + 3, *mlp_channels]
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.mlp = nn.Sequential(*blocks)

    def forward(self, xyz: Tensor, features: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if xyz.ndim != 3 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (batch, 3, points)")
        if features is not None and (
            features.ndim != 3
            or features.shape[0] != xyz.shape[0]
            or features.shape[2] != xyz.shape[2]
        ):
            raise ValueError("features must have shape (batch, channels, points)")

        xyz_points = xyz.transpose(1, 2).contiguous()
        feature_points = None if features is None else features.transpose(1, 2).contiguous()

        if self.group_all:
            centroids = xyz_points.mean(dim=1, keepdim=True)
            grouped_xyz = xyz_points.unsqueeze(1) - centroids.unsqueeze(2)
            grouped = grouped_xyz
            if feature_points is not None:
                grouped = torch.cat((grouped, feature_points.unsqueeze(1)), dim=-1)
        else:
            assert self.number is not None and self.radius is not None and self.samples is not None
            centroids, grouped = sample_and_group(
                self.number, self.radius, self.samples, xyz_points, feature_points
            )

        # Conv2d expects (B, channels, centroids, neighbours).
        encoded = self.mlp(grouped.permute(0, 3, 1, 2).contiguous())
        encoded = encoded.max(dim=-1).values
        return centroids.transpose(1, 2).contiguous(), encoded


class PointNet2Encoder(nn.Module):
    """Three-level single-scale grouping PointNet++ encoder."""

    def __init__(self, input_channels: int = 0) -> None:
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, input_channels, (64, 64, 128))
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, (128, 128, 256))
        self.sa3 = PointNetSetAbstraction(None, None, None, 256, (256, 512, 1024), group_all=True)

    def forward(self, xyz: Tensor, features: Tensor | None = None) -> Tensor:
        xyz1, features1 = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz1, features1)
        _, global_features = self.sa3(xyz2, features2)
        return global_features.squeeze(-1)


class PointNet2Classifier(nn.Module):
    """PointNet++ encoder followed by a small classification head."""

    def __init__(self, classes: int, input_channels: int = 0, dropout: float = 0.4) -> None:
        super().__init__()
        if classes <= 0:
            raise ValueError("classes must be positive")
        self.encoder = PointNet2Encoder(input_channels)
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, classes),
        )

    def forward(self, xyz: Tensor, features: Tensor | None = None) -> Tensor:
        return self.head(self.encoder(xyz, features))
