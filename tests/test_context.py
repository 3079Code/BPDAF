"""Equation-level tests for the manuscript's distance/energy descriptor."""

import math

import pytest
import torch

from bpdaf import DistanceEnergyContext


def test_radial_encoding_uses_cell_centers_and_farthest_range_corner():
    # An off-center, non-square map catches swapped axes and linspace endpoints.
    context = DistanceEnergyContext(
        bev_range=(-3.0, 1.0, 5.0, 7.0),
        num_frequencies=2,
        include_energy=False,
    )
    camera = torch.zeros(2, 3, 2, 4)
    lidar = torch.zeros(2, 5, 2, 4)
    actual = context(camera, lidar)
    x = torch.tensor([-2.0, 0.0, 2.0, 4.0])
    y = torch.tensor([2.5, 5.5])
    radius = torch.sqrt(y[:, None].square() + x[None, :].square()) / math.sqrt(74)
    expected = torch.stack(
        [
            radius,
            torch.sin(2 * math.pi * radius),
            torch.cos(2 * math.pi * radius),
            torch.sin(4 * math.pi * radius),
            torch.cos(4 * math.pi * radius),
        ]
    )
    assert context.output_channels == 5
    assert actual.shape == (2, 5, 2, 4)
    torch.testing.assert_close(actual[0], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[1], expected, rtol=1e-6, atol=1e-6)
    assert actual[:, 0].max() < 1  # Cell centers never reach the farthest corner.


def test_energy_matches_absolute_channel_mean_and_spatial_normalization():
    context = DistanceEnergyContext((-1, -1, 1, 1), include_distance=False)
    camera = torch.tensor([[[[1.0, -3.0], [5.0, -7.0]], [[-3.0, 1.0], [-7.0, 5.0]]]])
    lidar = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    actual = context(camera, lidar)
    expected_camera = torch.tensor([[2.0, 2.0], [6.0, 6.0]]) / 4.0
    expected_lidar = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2.5
    assert context.output_channels == 2
    torch.testing.assert_close(actual[0, 0], expected_camera)
    torch.testing.assert_close(actual[0, 1], expected_lidar)


def test_energy_is_scale_invariant_and_normalized_per_sample():
    torch.manual_seed(3)
    context = DistanceEnergyContext((-1, -1, 1, 1), include_distance=False)
    camera = torch.rand(1, 3, 3, 5) + 0.1
    lidar = torch.rand(1, 5, 3, 5) + 0.1
    single = context(camera, lidar)
    batched = context(torch.cat([camera, camera * 100]), torch.cat([lidar, lidar * 0.01]))
    torch.testing.assert_close(batched[0], single[0])
    torch.testing.assert_close(batched[1], single[0])


def test_energy_zero_floor_and_clipping_are_finite():
    context = DistanceEnergyContext((-1, -1, 1, 1), include_distance=False)
    camera = torch.zeros(1, 3, 3, 3)
    lidar = torch.zeros(1, 5, 3, 3)
    actual = context(camera, lidar)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0
    camera[:, :, 0, 0] = 9.0
    actual = context(camera, lidar)
    assert actual[0, 0, 0, 0] == 5.0
    assert actual[0, 0].sum() == 5.0
    assert torch.count_nonzero(actual[:, 1]) == 0
    tiny = torch.full_like(lidar, 1e-8)
    torch.testing.assert_close(context(tiny, tiny), torch.full((1, 2, 3, 3), 0.01))


def test_detachment_only_affects_energy_gradient_path():
    camera = torch.arange(1.0, 13.0).reshape(1, 2, 2, 3).requires_grad_()
    lidar = torch.arange(1.0, 7.0).reshape(1, 1, 2, 3).requires_grad_()
    detached = DistanceEnergyContext((-2, -3, 4, 5), detach_energy=True)
    attached = DistanceEnergyContext((-2, -3, 4, 5), detach_energy=False)
    assert not detached(camera, lidar).requires_grad
    cue = attached(camera, lidar)
    (cue[0, -2, 0, 0] + cue[0, -1, 1, 0]).backward()
    assert camera.grad is not None and camera.grad.abs().sum() > 0
    assert lidar.grad is not None and lidar.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_context_remains_finite(dtype):
    context = DistanceEnergyContext((-54, -54, 54, 54))
    camera = torch.zeros(2, 3, 3, 5, dtype=dtype)
    lidar = torch.zeros(2, 5, 3, 5, dtype=dtype)
    actual = context(camera, lidar)
    assert actual.shape == (2, 11, 3, 5)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, -2:]) == 0


@pytest.mark.parametrize(
    "camera_shape,lidar_shape",
    [((1, 3, 2, 4), (2, 5, 2, 4)), ((1, 3, 2, 4), (1, 5, 4, 2)), ((3, 2, 4), (5, 2, 4))],
)
def test_context_rejects_unaligned_inputs(camera_shape, lidar_shape):
    context = DistanceEnergyContext((-1, -1, 1, 1))
    with pytest.raises((ValueError, TypeError)):
        context(torch.zeros(camera_shape), torch.zeros(lidar_shape))
