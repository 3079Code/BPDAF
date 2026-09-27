"""Behavioral tests of BPDAF's preserved reference and trainable correction."""

import copy

import pytest
import torch
from torch import nn

from bpdaf import BPDAFFuser, ConvFuser

RANGE = (-3.0, -2.0, 5.0, 4.0)


def inputs(requires_grad=False):
    generator = torch.Generator().manual_seed(719)
    return [
        torch.randn(2, 3, 4, 6, generator=generator, requires_grad=requires_grad),
        torch.randn(2, 5, 4, 6, generator=generator, requires_grad=requires_grad),
    ]


def small_model(**kwargs):
    return BPDAFFuser(in_channels=(3, 5), out_channels=4, bev_range=RANGE, **kwargs)


@pytest.mark.parametrize("training", [False, True])
def test_initial_output_exactly_matches_reference_in_eval_and_train(training):
    torch.manual_seed(21)
    reference = ConvFuser(in_channels=(3, 5), out_channels=4)
    model = BPDAFFuser.from_reference(reference, bev_range=RANGE)
    reference.train(training)
    model.train(training)
    features = inputs()
    expected = reference(features)
    actual, aux = model.forward_with_aux(features)
    assert torch.equal(actual, expected)
    assert torch.equal(aux["reference"], expected)
    assert torch.count_nonzero(aux["residual"]) == 0
    assert torch.equal(aux["weights"], torch.full((2, 2, 4, 6), 0.5))
    torch.testing.assert_close(aux["weights"].sum(dim=1), torch.ones(2, 4, 6))
    assert aux["context"].shape == (2, 11, 4, 6)
    # A fresh clone gives identical outputs without unequal BN running updates.
    forward_model = copy.deepcopy(model)
    aux_model = copy.deepcopy(model)
    assert torch.equal(forward_model(features), aux_model.forward_with_aux(features)[0])
    for key, value in reference.state_dict().items():
        if "running_" in key or "num_batches_tracked" in key:
            assert torch.equal(value, model.state_dict()[key])


def test_module_construction_preserves_later_shared_layer_rng():
    torch.manual_seed(42)
    reference = ConvFuser(in_channels=(3, 5), out_channels=4)
    baseline_later_layer = nn.Linear(7, 5)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    model = small_model()
    adaptive_later_layer = nn.Linear(7, 5)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for key, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    for key, value in baseline_later_layer.state_dict().items():
        assert torch.equal(adaptive_later_layer.state_dict()[key], value)


def test_from_reference_has_independent_parameters_and_preserves_rng():
    reference = ConvFuser(in_channels=(3, 5), out_channels=4)
    state_before = torch.random.get_rng_state().clone()
    model = BPDAFFuser.from_reference(reference, bev_range=RANGE)
    assert torch.equal(torch.random.get_rng_state(), state_before)
    model_parameters = dict(model.named_parameters())
    for key, parameter in reference.named_parameters():
        copied = model_parameters[key]
        assert copied.data_ptr() != parameter.data_ptr()
        assert torch.equal(copied, parameter)
    with torch.no_grad():
        model_parameters["0.weight"].add_(1)
    assert not torch.equal(
        model_parameters["0.weight"], dict(reference.named_parameters())["0.weight"]
    )


def test_baseline_state_loading_accepts_only_the_reference_keys():
    reference = ConvFuser(in_channels=(3, 5), out_channels=4)
    model = small_model()
    model.load_reference_state_dict(reference.state_dict())
    for key, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    missing = dict(reference.state_dict())
    missing.pop("0.weight")
    with pytest.raises((RuntimeError, ValueError)):
        model.load_reference_state_dict(missing)
    unexpected = dict(reference.state_dict())
    unexpected["unexpected.weight"] = torch.zeros(1)
    with pytest.raises((RuntimeError, ValueError)):
        model.load_reference_state_dict(unexpected)


def test_zero_initialization_blocks_only_upstream_residual_gradients():
    torch.manual_seed(25)
    model = small_model().eval()
    features = inputs(requires_grad=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    model(features).square().mean().backward()
    adaptive = model.adaptive
    assert adaptive.output_projection.weight.grad.abs().sum() > 0
    for module in [adaptive.camera_projection, adaptive.lidar_projection, adaptive.gate]:
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0
    assert dict(model.named_parameters())["0.weight"].grad.abs().sum() > 0
    assert all(feature.grad is not None and feature.grad.abs().sum() > 0 for feature in features)

    # One update opens the correction path; the final gate layer can now learn.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(features).square().mean().backward()
    assert adaptive.camera_projection.weight.grad.abs().sum() > 0
    assert adaptive.lidar_projection.weight.grad.abs().sum() > 0
    assert adaptive.gate[2].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(adaptive.gate[0].weight.grad) == 0

    # Its initially zero final kernel delays gradients into the hidden gate.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(features).square().mean().backward()
    assert adaptive.gate[0].weight.grad.abs().sum() > 0


def test_learned_residual_can_change_reference_output():
    model = small_model().eval()
    with torch.no_grad():
        model.adaptive.output_projection.weight.fill_(0.05)
    actual, aux = model.forward_with_aux(inputs())
    torch.testing.assert_close(actual, torch.relu(aux["reference"] + aux["residual"]))
    assert torch.count_nonzero(aux["residual"]) > 0
    assert not torch.equal(actual, aux["reference"])


def test_canonical_architecture_and_parameter_budget():
    model = BPDAFFuser(bev_range=(-54, -54, 54, 54))
    adaptive = model.adaptive
    assert adaptive.camera_projection.weight.shape == (256, 80, 1, 1)
    assert adaptive.lidar_projection.weight.shape == (256, 256, 1, 1)
    assert adaptive.output_projection.weight.shape == (256, 512, 3, 3)
    for layer in [
        adaptive.camera_projection,
        adaptive.lidar_projection,
        adaptive.output_projection,
    ]:
        assert isinstance(layer, nn.Conv2d)
        assert layer.bias is None
        assert layer.groups == 1
    assert adaptive.output_projection.padding == (1, 1)
    assert adaptive.gate[0].weight.shape == (32, 11, 1, 1)
    assert adaptive.gate[2].weight.shape == (2, 32, 1, 1)
    assert isinstance(adaptive.gate[1], nn.ReLU)
    assert torch.count_nonzero(adaptive.gate[2].weight) == 0
    assert torch.count_nonzero(adaptive.gate[2].bias) == 0
    assert torch.count_nonzero(adaptive.output_projection.weight) == 0
    added = sum(parameter.numel() for parameter in adaptive.parameters())
    assert 1_266_082 <= added <= 1_266_114  # First gate bias is unspecified by the paper.
    assert round(added / 1e6, 2) == 1.27


@pytest.mark.parametrize(
    "include_distance,include_energy,channels",
    [(True, True, 11), (False, True, 2), (True, False, 9)],
)
def test_cue_ablations_have_matching_gate_channels(include_distance, include_energy, channels):
    model = small_model(include_distance=include_distance, include_energy=include_energy)
    output, aux = model.forward_with_aux(inputs())
    assert output.shape == (2, 4, 4, 6)
    assert aux["context"].shape[1] == channels
    assert model.adaptive.gate[0].in_channels == channels


def test_sigmoid_ablation_uses_independent_unbounded_sum_gates():
    model = small_model(gate_mode="sigmoid")
    with torch.no_grad():
        model.adaptive.gate[2].bias.copy_(torch.tensor([1.0, 2.0]))
    _, aux = model.forward_with_aux(inputs())
    expected = torch.sigmoid(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(aux["weights"][0, :, 0, 0], expected)
    assert (aux["weights"].sum(dim=1) > 1).all()


@pytest.mark.parametrize(
    "bad_inputs",
    [
        [torch.zeros(2, 3, 4, 6)],
        [torch.zeros(2, 4, 4, 6), torch.zeros(2, 5, 4, 6)],
        [torch.zeros(2, 3, 4, 6), torch.zeros(1, 5, 4, 6)],
        [torch.zeros(2, 3, 4, 6), torch.zeros(2, 5, 6, 4)],
        [torch.zeros(3, 4, 6), torch.zeros(5, 4, 6)],
    ],
)
def test_fusion_rejects_malformed_or_unaligned_features(bad_inputs):
    with pytest.raises((TypeError, ValueError)):
        small_model()(bad_inputs)
