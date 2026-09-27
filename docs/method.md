# Method and implementation contract

This document describes the implementation of the BPDAF module in Sections 3.1–3.5 of the manuscript. Detector-specific preprocessing, encoders, heads, and losses are outside the standalone package.

## Inputs and coordinate convention

The input order is **camera, LiDAR**. The two floating-point tensors have shapes `(N, Cc, H, W)` and `(N, Cl, H, W)` and share batch size, spatial dimensions, device, and dtype. They must already represent the same BEV coordinates; this module does not align sensors or transform camera images to BEV.

The range argument is `(xmin, ymin, xmax, ymax)` in metres. Column `j` follows x and row `i` follows y. The origin `(0, 0)` is the sensor/ego origin used by the fused feature grid. For cell centers,

```text
x_j = xmin + (j + 1/2) (xmax - xmin) / W
y_i = ymin + (i + 1/2) (ymax - ymin) / H
rmax = max(sqrt(x² + y²) for the four range corners)
r_ij = sqrt(x_j² + y_i²) / rmax
```

Using pixel indices as distances or silently transposing H and W changes the spatial cue. For an asymmetric grid, supply its actual metric bounds, including any displacement of the sensor origin relative to the grid center.

## Reference path

The reference uses a bias-free 3 × 3 convolution with padding one, batch normalization, and ReLU:

```text
F_b = ReLU(BN(Conv3x3(concat(F_c, F_l))))
```

Its output channel count is 256 for the manuscript's default `(80, 256)` input channels. All reference parameters remain trainable. Batch-normalization running statistics are retained when copying a reference with `from_reference`.

## Distance and energy context

With K = 4 and k = 0, …, K − 1, the radial descriptor consists of

```text
e_d(r) = [r, {sin(2π · 2^k · r), cos(2π · 2^k · r)} for each k]
```

There are `1 + 2K = 9` fixed radial channels. For each modality m, the default energy cue is

```text
a_m[n, i, j] = mean_c(abs(detach(F_m[n, c, i, j])))
e_m[n, i, j] = clip(
    a_m[n, i, j] / max(mean_(u,v)(a_m[n, u, v]), 1e-6),
    0,
    5,
)
g = concat(e_d, e_c, e_l)
```

Spatial normalization is separate for each sample and modality. The context has 11 channels. The energy maps contain relative activation magnitudes, not calibrated confidence. Detaching the cue prevents gradients through **cue extraction**; input features still receive gradients through the reference and projected-feature paths.

This implementation uses the channel order `r, sin(2πr), cos(2πr), sin(4πr), cos(4πr), ...`, followed by camera energy and LiDAR energy. The manuscript does not specify a serialized Fourier-channel ordering. A newly trained gate is insensitive to a consistent ordering choice, but an externally trained gate checkpoint is not. Confirm the channel order and all numerical conventions before importing non-reference weights from another implementation.

For float16 and bfloat16 inputs, context construction and energy reductions use float32 intermediates, then return the input dtype. This numerical safeguard belongs to the new implementation; equivalence to an unavailable original mixed-precision implementation has not been established.

## Competitive residual

Two 1 × 1 gate convolutions form an 11 → 32 → 2 mapping with an intermediate ReLU. Their two logits are normalized over the modality dimension:

```text
z = gate2(ReLU(gate1(g)))
[alpha_c, alpha_l] = softmax(z, dim=modality)
alpha_c + alpha_l = 1
P_c = Conv1x1_c(F_c)
P_l = Conv1x1_l(F_l)
delta = Conv3x3_delta(concat(alpha_c * P_c, alpha_l * P_l))
F_out = ReLU(F_b + delta)
```

The separate modality projections are bias-free linear convolutions without normalization or activation. Each projects to `out_channels`, so their weighted concatenation has `2 * out_channels` channels. The residual output convolution is bias-free with a 3 × 3 kernel and padding one.

Weights are broadcast across projected channels at each cell. They control the residual branch; the retained reference path also consumes both modalities. Thus a camera weight of 0.8 does not mean that 80% of the complete detector output comes from the camera.

## Initialization and gradients

The residual output kernel and the final gate kernel/bias start at zero. Therefore:

```text
delta = 0
alpha_c = alpha_l = 0.5
F_out = ReLU(F_b) = F_b
```

The last equality uses the reference's terminal ReLU. Equal modality weights alone would not make the residual zero. Exact reference preservation depends on the zero output kernel and the nonnegative reference output.

The double zero initialization also creates a staged gradient path:

1. On the initial backward pass, the residual output convolution can receive a gradient. The zero output kernel blocks gradients to the projections and gate.
2. Once that kernel changes, gradients can reach the projections and final gate convolution, subject to the loss, activations, and modality features.
3. The first gate convolution receives a nonzero signal only after the final gate weights move away from zero; a changed gate bias alone does not unblock this path.

These are structural statements about backpropagation. They do not guarantee that any particular loss or batch produces nonzero gradients, and they do not promise lower training variance.

Branch construction preserves the random-number-generator state around initialization of the additional layers. This prevents the extra branch from advancing the RNG stream used to initialize subsequent shared layers. It is a construction-time control, not a guarantee of identical training trajectories, data-loader order, CUDA behavior, or results across devices and PyTorch versions.

In this release all new convolutions are constructed on CPU, and the additional
branch is created inside `torch.random.fork_rng(devices=[])`. The CPU RNG is
restored; branch construction uses no CUDA random draws. Modules can then be
moved to a device with `.to(...)`. The first and last gate convolutions have
biases. Random kernels use the PyTorch `Conv2d` defaults before the specified
zero initialization. A newly constructed reference uses BatchNorm epsilon
`1e-5` and momentum `0.1`; `from_reference` retains an existing reference's
settings. These choices resolve details not fully specified in the manuscript.

## API and checkpoint boundary

`BPDAFFuser.forward` returns only `F_out`. `forward_with_aux` returns `(F_out, aux)`, where `aux` contains `reference`, `residual`, `weights`, and `context`. The auxiliary tensors are intended for inspection; retaining them across iterations also retains computation graphs unless the caller detaches them.

`BPDAFFuser.from_reference` copies a compatible reference fuser rather than sharing its parameter storage. Reference parameter/buffer compatibility does not establish compatibility with an entire external detector checkpoint. Follow [integration.md](integration.md) to select the correct module and validate loading.

Geometry and constructor options must be saved alongside the state dictionary. Tensor weights alone do not record the BEV range or establish the intended coordinate convention. Recreate the same configuration before loading a full BPDAF checkpoint.

## Controlled switches

The constructor exposes `include_distance`, `include_energy`, `detach_energy`, `gate_mode`, and `zero_init_residual`. Defaults implement the full mechanism. Disabling a cue removes its context channels; choosing `gate_mode="sigmoid"` removes complementary normalization; disabling zero residual initialization removes the initial reference-equality guarantee. These switches allow new controlled experiments, but their existence does not reproduce the manuscript's ablation scores. The reference path is always retained by this class; it does not implement the manuscript's direct-replacement comparison variant.

## Experimental boundary

The synthetic example is deliberately independent of any dataset or detector loss. In the manuscript, BPDAF uses the existing detection objective: dense heatmap classification, decoder classification, and weighted box regression, with no gate target or auxiliary distance loss. Reproducing detection results requires those detector components and the complete data protocol; see [reproduction.md](reproduction.md).
