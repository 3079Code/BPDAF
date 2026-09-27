"""Adapter contract tests with a mocked registry, not a full detector run."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

from bpdaf import ConvFuser


@pytest.fixture
def adapter_class(monkeypatch):
    registered = {}

    class Registry:
        @staticmethod
        def register_module(name):
            def register(cls):
                registered[name] = cls
                return cls

            return register

    package = types.ModuleType("mmdet3d")
    registry_module = types.ModuleType("mmdet3d.registry")
    registry_module.MODELS = Registry()
    monkeypatch.setitem(sys.modules, "mmdet3d", package)
    monkeypatch.setitem(sys.modules, "mmdet3d.registry", registry_module)
    plugin = Path(__file__).resolve().parents[1] / "integrations/mmdetection3d/bpdaf_plugin.py"
    spec = importlib.util.spec_from_file_location("_bpdaf_test_plugin", plugin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert registered == {"BPDAFFuser": module.MMDetection3DBPDAF}
    return registered["BPDAFFuser"]


def test_adapter_geometry_matches_host_xy_layout_without_transposing(adapter_class):
    # Host H follows physical x; host W follows physical y. The unequal range
    # and unequal axis lengths make an unnoticed axis swap observable.
    physical_range = (2.0, -8.0, 14.0, 2.0)
    model = adapter_class(in_channels=(3, 5), out_channels=4, bev_range=physical_range).eval()
    generator = torch.Generator().manual_seed(64)
    camera = torch.randn(2, 3, 3, 5, generator=generator)
    lidar = torch.randn(2, 5, 3, 5, generator=generator)
    output, aux = model.forward_with_aux([camera, lidar])
    physical_x = torch.tensor([4.0, 8.0, 12.0])
    physical_y = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0])
    expected_radius = torch.sqrt(
        physical_x[:, None].square() + physical_y[None, :].square()
    ) / math.sqrt(14**2 + 8**2)
    torch.testing.assert_close(aux["context"][0, 0], expected_radius)
    torch.testing.assert_close(aux["context"][1, 0], expected_radius)
    assert output.shape == (2, 4, 3, 5)
    assert model.physical_bev_range == physical_range
    assert model.adaptive.context.bev_range == (-8.0, 2.0, 2.0, 14.0)


def test_adapter_keeps_numeric_reference_keys_and_reference_output(adapter_class):
    reference = ConvFuser(in_channels=(3, 5), out_channels=4).eval()
    model = adapter_class(in_channels=(3, 5), out_channels=4, bev_range=(2, -8, 14, 2)).eval()
    model.load_reference_state_dict(reference.state_dict())
    camera = torch.randn(2, 3, 3, 5)
    lidar = torch.randn(2, 5, 3, 5)
    assert torch.equal(model([camera, lidar]), reference([camera, lidar]))
    for name, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[name], value)


def test_adapter_rejects_incomplete_physical_range(adapter_class):
    with pytest.raises(ValueError):
        adapter_class(bev_range=(0, 1, 2))
