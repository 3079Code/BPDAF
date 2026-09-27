"""Input checks shared by the context encoder and fusion layers."""

from collections.abc import Sequence

import torch
from torch import Tensor


def channel_pair(in_channels: Sequence[int]) -> tuple[int, int]:
    if not isinstance(in_channels, (list, tuple)) or len(in_channels) != 2:
        raise ValueError("in_channels must contain camera and LiDAR channel counts")
    if any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in in_channels):
        raise ValueError("channel counts must be positive integers")
    return in_channels[0], in_channels[1]


def validate_pair(camera: Tensor, lidar: Tensor, channels: tuple[int, int] | None = None) -> None:
    for name, value in (("camera", camera), ("lidar", lidar)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4 or any(size <= 0 for size in value.shape):
            raise ValueError(f"{name} must have nonempty shape [B, C, H, W]")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must have a floating-point dtype")
    if (camera.shape[0], *camera.shape[2:]) != (lidar.shape[0], *lidar.shape[2:]):
        raise ValueError("camera and lidar must share batch size and BEV spatial shape")
    if camera.device != lidar.device or camera.dtype != lidar.dtype:
        raise ValueError("camera and lidar must have the same device and dtype")
    if channels is not None and (camera.shape[1], lidar.shape[1]) != channels:
        raise ValueError(f"expected camera/LiDAR channels {channels}")


def unpack_features(inputs: Sequence[Tensor], channels: tuple[int, int]) -> tuple[Tensor, Tensor]:
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        raise ValueError("inputs must be [camera_features, lidar_features]")
    camera, lidar = inputs
    validate_pair(camera, lidar, channels)
    return camera, lidar
