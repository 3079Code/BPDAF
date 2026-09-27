"""Reference-preserving BEV fusion from manuscript equations (1), (7)--(12)."""

import copy
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from ._validation import channel_pair, unpack_features
from .context import DistanceEnergyContext


class ConvFuser(nn.Sequential):
    """Camera-first concatenation followed by Conv3x3, BatchNorm and ReLU.

    Numeric layer names are intentional: existing sequential ConvFuser
    checkpoint keys (``0.weight``, ``1.running_mean``, etc.) remain valid.
    Construction happens on CPU; move the layer with ``.to(...)`` afterward.
    """

    def __init__(self, in_channels: Sequence[int] = (80, 256), out_channels: int = 256) -> None:
        channels = channel_pair(in_channels)
        if isinstance(out_channels, bool) or not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        super().__init__(
            nn.Conv2d(sum(channels), out_channels, 3, padding=1, bias=False, device="cpu"),
            nn.BatchNorm2d(out_channels, device="cpu"),
            nn.ReLU(inplace=True),
        )
        self.in_channels = channels
        self.out_channels = out_channels

    def _reference_forward(self, camera: Tensor, lidar: Tensor) -> Tensor:
        # Do not iterate over self: subclasses register an additional branch.
        return self[2](self[1](self[0](torch.cat((camera, lidar), dim=1))))

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        camera, lidar = unpack_features(inputs, self.in_channels)
        return self._reference_forward(camera, lidar)


class _AdaptiveResidual(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int,
        bev_range: Sequence[float],
        num_frequencies: int,
        gate_hidden_channels: int,
        energy_eps: float,
        energy_clip: float,
        include_distance: bool,
        include_energy: bool,
        detach_energy: bool,
        gate_mode: str,
        zero_init_residual: bool,
    ) -> None:
        super().__init__()
        if gate_mode not in ("softmax", "sigmoid"):
            raise ValueError("gate_mode must be 'softmax' or 'sigmoid'")
        if (
            isinstance(gate_hidden_channels, bool)
            or not isinstance(gate_hidden_channels, int)
            or gate_hidden_channels <= 0
        ):
            raise ValueError("gate_hidden_channels must be a positive integer")
        self.context = DistanceEnergyContext(
            bev_range,
            num_frequencies=num_frequencies,
            eps=energy_eps,
            clip=energy_clip,
            include_distance=include_distance,
            include_energy=include_energy,
            detach_energy=detach_energy,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(self.context.output_channels, gate_hidden_channels, 1, device="cpu"),
            nn.ReLU(inplace=False),
            nn.Conv2d(gate_hidden_channels, 2, 1, device="cpu"),
        )
        self.camera_projection = nn.Conv2d(
            in_channels[0], out_channels, 1, bias=False, device="cpu"
        )
        self.lidar_projection = nn.Conv2d(in_channels[1], out_channels, 1, bias=False, device="cpu")
        self.output_projection = nn.Conv2d(
            2 * out_channels, out_channels, 3, padding=1, bias=False, device="cpu"
        )
        self.gate_mode = gate_mode
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        if zero_init_residual:
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, camera: Tensor, lidar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        context = self.context(camera, lidar)
        logits = self.gate(context)
        weights = logits.softmax(dim=1) if self.gate_mode == "softmax" else logits.sigmoid()
        camera_part = self.camera_projection(camera) * weights[:, 0:1]
        lidar_part = self.lidar_projection(lidar) * weights[:, 1:2]
        residual = self.output_projection(torch.cat((camera_part, lidar_part), dim=1))
        return residual, weights, context


class BPDAFFuser(ConvFuser):
    """Baseline-Preserving Distance-Aware Adaptive Fusion.

    Defaults implement the manuscript module. For finite inputs, initialization
    reproduces the reference ConvFuser exactly (not a frozen reference model).
    The reference parameters remain trainable.  Optional cue/gating switches
    are controlled ablations; omitting a cue removes its context channels.

    Added layers are created on CPU inside a torch RNG fork, preserving draws
    used by later shared layers. Their random initialization uses PyTorch's
    Conv2d defaults, followed by zeroing the two specified output convolutions.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (80, 256),
        out_channels: int = 256,
        bev_range: Sequence[float] = (-54.0, -54.0, 54.0, 54.0),
        num_frequencies: int = 4,
        gate_hidden_channels: int = 32,
        energy_eps: float = 1e-6,
        energy_clip: float = 5.0,
        include_distance: bool = True,
        include_energy: bool = True,
        detach_energy: bool = True,
        gate_mode: str = "softmax",
        zero_init_residual: bool = True,
        *,
        reference: nn.Sequential | None = None,
    ) -> None:
        channels = channel_pair(in_channels)
        if reference is None:
            super().__init__(channels, out_channels)
        else:
            self._validate_reference(reference, channels, out_channels)
            cloned = copy.deepcopy(reference)
            nn.Sequential.__init__(self, *list(cloned.children()))
            self.in_channels = channels
            self.out_channels = out_channels
            self.train(reference.training)
        with torch.random.fork_rng(devices=[]):
            self.adaptive = _AdaptiveResidual(
                channels,
                out_channels,
                bev_range,
                num_frequencies,
                gate_hidden_channels,
                energy_eps,
                energy_clip,
                include_distance,
                include_energy,
                detach_energy,
                gate_mode,
                zero_init_residual,
            )
        self.adaptive.to(device=self[0].weight.device, dtype=self[0].weight.dtype)
        self.adaptive.train(self.training)

    @staticmethod
    def _validate_reference(
        reference: nn.Sequential, channels: tuple[int, int], out_channels: int
    ) -> None:
        if not isinstance(reference, nn.Sequential) or list(reference._modules) != ["0", "1", "2"]:
            raise ValueError("reference must be a three-layer numeric Sequential Conv/BN/ReLU")
        conv, norm, activation = list(reference.children())
        if not isinstance(conv, nn.Conv2d) or not isinstance(norm, nn.BatchNorm2d):
            raise ValueError("reference must contain Conv2d followed by BatchNorm2d")
        if not isinstance(activation, nn.ReLU):
            raise ValueError("reference must end in ReLU to preserve its initial output")
        if (
            conv.in_channels != sum(channels)
            or conv.out_channels != out_channels
            or conv.kernel_size != (3, 3)
            or conv.stride != (1, 1)
            or conv.padding != (1, 1)
            or conv.dilation != (1, 1)
            or conv.groups != 1
            or conv.bias is not None
            or norm.num_features != out_channels
        ):
            raise ValueError("reference shape or convolution settings do not match ConvFuser")

    @classmethod
    def from_reference(
        cls, reference: nn.Sequential, bev_range: Sequence[float], **kwargs
    ) -> "BPDAFFuser":
        """Clone a compatible reference, including its BN buffers and settings.

        The original module is not mutated or shared. The reference must expose
        ``in_channels`` (camera, LiDAR) and ``out_channels`` like ConvFuser.
        """
        if not hasattr(reference, "in_channels") or not hasattr(reference, "out_channels"):
            raise ValueError("reference must expose in_channels and out_channels")
        return cls(
            in_channels=reference.in_channels,
            out_channels=reference.out_channels,
            bev_range=bev_range,
            reference=reference,
            **kwargs,
        )

    def load_reference_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load only the complete reference state, rejecting unmatched keys.

        Pass the fuser-local mapping, without a detector prefix. This method
        deliberately does not load an arbitrary partial detector checkpoint.
        Use normal ``load_state_dict`` for a complete BPDAF checkpoint.
        """
        expected = {key for key in self.state_dict() if not key.startswith("adaptive.")}
        supplied = set(state_dict)
        if supplied != expected:
            raise ValueError(
                f"reference state mismatch: missing={sorted(expected - supplied)}, "
                f"unexpected={sorted(supplied - expected)}"
            )
        # Check all shapes before loading so a malformed state cannot partially
        # overwrite the module before PyTorch reports a mismatch.
        own_state = self.state_dict()
        for key, value in state_dict.items():
            if not isinstance(value, Tensor) or value.shape != own_state[key].shape:
                raise ValueError(f"reference state has an invalid tensor shape at {key}")
        self.load_state_dict(state_dict, strict=False)

    def forward_with_aux(self, inputs: Sequence[Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Return output and differentiable diagnostic tensors.

        Callers storing diagnostics across steps should detach them to avoid
        retaining autograd graphs. Weights describe residual allocation, not
        calibrated sensor confidence or full-detector modality contributions.
        """
        camera, lidar = unpack_features(inputs, self.in_channels)
        reference = self._reference_forward(camera, lidar)
        residual, weights, context = self.adaptive(camera, lidar)
        output = torch.relu(reference + residual)
        return output, {
            "reference": reference,
            "residual": residual,
            "weights": weights,
            "context": context,
        }

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        output, _ = self.forward_with_aux(inputs)
        return output
