"""Persistence, mixed precision, and checkpoint failure behavior."""

import io

import pytest
import torch

from bpdaf import BPDAFFuser, ConvFuser

RANGE = (-2.0, -4.0, 6.0, 8.0)


def features():
    generator = torch.Generator().manual_seed(91)
    return [
        torch.randn(2, 3, 4, 5, generator=generator),
        torch.randn(2, 5, 4, 5, generator=generator),
    ]


def test_full_state_dict_round_trip_with_safe_tensor_loading():
    torch.manual_seed(4)
    original = BPDAFFuser(in_channels=(3, 5), out_channels=4, bev_range=RANGE).eval()
    # Exercise learned, nonuniform gating and a nonzero correction, not only
    # a state whose output accidentally reduces to the original baseline.
    with torch.no_grad():
        original.adaptive.output_projection.weight.normal_(0, 0.05)
        original.adaptive.gate[2].weight.normal_(0, 0.1)
        original.adaptive.gate[2].bias.copy_(torch.tensor([0.2, -0.3]))
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    saved_state = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = BPDAFFuser(in_channels=(3, 5), out_channels=4, bev_range=RANGE).eval()
    restored.load_state_dict(saved_state, strict=True)
    actual, actual_aux = restored.forward_with_aux(features())
    expected, expected_aux = original.forward_with_aux(features())
    assert torch.equal(actual, expected)
    for name in ("reference", "residual", "weights", "context"):
        assert torch.equal(actual_aux[name], expected_aux[name])
    assert torch.count_nonzero(actual_aux["residual"]) > 0


def test_cpu_bfloat16_autocast_preserves_initial_output_and_finite_backward():
    reference = ConvFuser(in_channels=(3, 5), out_channels=4)
    model = BPDAFFuser.from_reference(reference, bev_range=RANGE)
    inputs = [tensor.requires_grad_() for tensor in features()]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = reference(inputs)
        actual, aux = model.forward_with_aux(inputs)
        loss = actual.float().square().mean()
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(aux["residual"]) == 0
    assert torch.isfinite(actual).all()
    assert all(torch.isfinite(value).all() for value in aux.values())
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.adaptive.output_projection.weight.grad.abs().sum() > 0
    for value in inputs:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_from_reference_preserves_eval_and_custom_batch_norm_configuration():
    reference = ConvFuser(in_channels=(3, 5), out_channels=4).eval()
    reference[1].eps = 0.003
    reference[1].momentum = 0.17
    with torch.no_grad():
        reference[1].running_mean.copy_(torch.tensor([0.4, -0.7, 1.2, 0.3]))
        reference[1].running_var.copy_(torch.tensor([0.8, 1.3, 0.5, 2.0]))
        reference[1].weight.copy_(torch.tensor([0.7, 1.2, 0.6, 1.8]))
        reference[1].bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        reference[1].num_batches_tracked.fill_(17)
    model = BPDAFFuser.from_reference(reference, bev_range=RANGE)
    assert not model.training
    assert all(not child.training for child in model.modules())
    assert model[1].eps == reference[1].eps
    assert model[1].momentum == reference[1].momentum
    assert model[1].running_mean.data_ptr() != reference[1].running_mean.data_ptr()
    assert torch.equal(model(features()), reference(features()))
    assert model[1].num_batches_tracked.item() == 17


def test_malformed_reference_shape_is_rejected_before_any_parameter_is_changed():
    reference = ConvFuser(in_channels=(3, 5), out_channels=4)
    model = BPDAFFuser(in_channels=(3, 5), out_channels=4, bev_range=RANGE)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    malformed = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    malformed["0.weight"].fill_(99)  # A valid earlier key must not be loaded.
    malformed["1.running_var"] = torch.ones(5)  # Invalid later key.
    with pytest.raises((ValueError, RuntimeError)):
        model.load_reference_state_dict(malformed)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), f"partially mutated {name}"
