"""Distance and feature-energy context from manuscript equations (2)--(6)."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ._validation import validate_pair


class DistanceEnergyContext(nn.Module):
    """Encode radius and per-sample, per-modality relative feature energy.

    ``bev_range`` is ``(xmin, ymin, xmax, ymax)`` in the sensor frame.
    Tensor rows follow y and columns follow x.  The radial channel order is
    ``r, sin(2*pi*r), cos(2*pi*r), sin(4*pi*r), cos(4*pi*r), ...``.
    Energy channels, when enabled, follow in camera then LiDAR order.

    Half and bfloat16 reductions are evaluated in float32; the resulting
    context is cast back to the input dtype. The range and conventions must
    match the feature producer. They cannot be inferred from tensor size.
    """

    def __init__(
        self,
        bev_range: Sequence[float],
        num_frequencies: int = 4,
        eps: float = 1e-6,
        clip: float = 5.0,
        include_distance: bool = True,
        include_energy: bool = True,
        detach_energy: bool = True,
    ) -> None:
        super().__init__()
        if len(bev_range) != 4:
            raise ValueError("bev_range must be (xmin, ymin, xmax, ymax)")
        self.bev_range = tuple(float(value) for value in bev_range)
        xmin, ymin, xmax, ymax = self.bev_range
        if not all(math.isfinite(value) for value in self.bev_range):
            raise ValueError("bev_range must be finite")
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("BEV range minima must be strictly less than maxima")
        if isinstance(num_frequencies, bool) or not isinstance(num_frequencies, int):
            raise ValueError("num_frequencies must be a nonnegative integer")
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be a nonnegative integer")
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(clip) or clip <= 0:
            raise ValueError("eps and clip must be finite and positive")
        if not include_distance and not include_energy:
            raise ValueError("at least one context cue must be enabled")
        self.num_frequencies = num_frequencies
        self.eps = float(eps)
        self.clip = float(clip)
        self.include_distance = include_distance
        self.include_energy = include_energy
        self.detach_energy = detach_energy
        self.rmax = math.hypot(max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax)))
        self.output_channels = (1 + 2 * num_frequencies if include_distance else 0) + (
            2 if include_energy else 0
        )

    @staticmethod
    def _work_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    def _radial_code(self, feature: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        xmin, ymin, xmax, ymax = self.bev_range
        dtype = self._work_dtype(feature.dtype)
        x = xmin + (torch.arange(width, device=feature.device, dtype=dtype) + 0.5) * (
            (xmax - xmin) / width
        )
        y = ymin + (torch.arange(height, device=feature.device, dtype=dtype) + 0.5) * (
            (ymax - ymin) / height
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = (torch.sqrt(xx.square() + yy.square()) / self.rmax).clamp(0.0, 1.0)
        channels = [radius]
        for index in range(self.num_frequencies):
            angle = (2.0 * math.pi * 2**index) * radius
            channels.extend((angle.sin(), angle.cos()))
        return torch.stack(channels, dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def _energy(self, feature: Tensor) -> Tensor:
        source = feature.detach() if self.detach_energy else feature
        source = source.to(dtype=self._work_dtype(source.dtype))
        activation = source.abs().mean(dim=1, keepdim=True)
        denominator = activation.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return (activation / denominator).clamp(min=0.0, max=self.clip)

    def forward(self, camera: Tensor, lidar: Tensor) -> Tensor:
        validate_pair(camera, lidar)
        parts = []
        if self.include_distance:
            parts.append(self._radial_code(camera))
        if self.include_energy:
            parts.extend((self._energy(camera), self._energy(lidar)))
        return torch.cat(parts, dim=1).to(dtype=camera.dtype)

    def extra_repr(self) -> str:
        return (
            f"bev_range={self.bev_range}, channels={self.output_channels}, "
            f"detach_energy={self.detach_energy}"
        )
