# BPDAF

**Baseline-Preserving Distance-Aware Adaptive Fusion for LiDAR-Camera 3D Object Detection**

Jiaxin Yang, Yongbiao Li, Zhanlin Cao, Long Chen, and Jinglong Wang

[中文说明](README.zh-CN.md) · [Method](docs/method.md) · [Integration](docs/integration.md) · [Validation](docs/validation.md) · [Reproduction status](docs/reproduction.md)

BPDAF adds a distance- and feature-energy-conditioned residual to aligned camera and LiDAR bird's-eye-view (BEV) features. The retained convolutional fusion path remains trainable. A zero-initialized residual output convolution makes the initial fused output equal to that of the reference path.

## Release scope

This repository is a **fresh, paper-guided PyTorch reimplementation**. It provides the fusion module, tests, a synthetic training example, and an optional detector-integration adapter. It is not the original code used to obtain the manuscript's experimental results. No new nuScenes or KITTI benchmark reproduction is claimed.

The release does not include a complete detector, dataset files, pretrained checkpoints, or the complete experiment configurations. The known reproduction gaps are recorded in [docs/reproduction.md](docs/reproduction.md). The core package needs only PyTorch; CUDA extensions and a detector framework are not required for its unit tests.

## Installation

Use Python 3.10 or later and PyTorch 2.1 or later. Install the appropriate PyTorch build for your system first, using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). In this repository's root directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python examples/smoke_train.py
```

To also exercise checkpoint saving and exact reload, run
`python examples/smoke_train.py --output runs/smoke.pt`.

The example trains on synthetic feature tensors to demonstrate gradient flow; its loss is not the paper's detection objective and its output is not a detection benchmark.

CPU CI is configured for Python 3.10 / PyTorch 2.1.2 and Python 3.12 / PyTorch 2.6.0. This matrix does not certify every version allowed by the dependency declaration or any external detection stack.

## Quick start

```python
import torch

from bpdaf import BPDAFFuser

# Features must already be aligned in the same BEV coordinate frame.
# Input order is camera, then LiDAR. The range order is xmin, ymin, xmax, ymax.
fuser = BPDAFFuser(
    in_channels=(80, 256),
    out_channels=256,
    bev_range=(-54.0, -54.0, 54.0, 54.0),
)
camera = torch.randn(2, 80, 24, 24)
lidar = torch.randn(2, 256, 24, 24)

output, auxiliary = fuser.forward_with_aux([camera, lidar])
assert output.shape == (2, 256, 24, 24)
assert torch.count_nonzero(auxiliary["residual"]) == 0
torch.testing.assert_close(output, auxiliary["reference"], rtol=0, atol=0)
torch.testing.assert_close(auxiliary["weights"], torch.full_like(auxiliary["weights"], 0.5))
```

`fuser([camera, lidar])` returns only the fused output. `forward_with_aux` additionally returns the reference tensor, residual, two-channel modality weights, and context tensor. The small grid above is for demonstration; the manuscript uses a 180 × 180 nuScenes fusion grid over the stated range.

### Start from a reference fuser

```python
from bpdaf import BPDAFFuser, ConvFuser

reference = ConvFuser(in_channels=(80, 256), out_channels=256)
# Load the reference's own compatible state dictionary here, if available.
fuser = BPDAFFuser.from_reference(
    reference,
    bev_range=(-54.0, -54.0, 54.0, 54.0),
)
```

`from_reference` copies a compatible Conv2d → BatchNorm2d → ReLU reference, including its learned parameters and normalization state. It does not share its parameter storage with the supplied module. A full detector checkpoint needs the explicit mapping and compatibility checks described in [the integration guide](docs/integration.md); a checkpoint from an arbitrary framework is not automatically interchangeable.

## Mechanism

| Component | Default used for the paper-guided module |
| --- | --- |
| Reference | 3 × 3 convolution → batch normalization → ReLU |
| Spatial cue | Cell-center radius normalized by the farthest BEV-range corner |
| Radial encoding | Radius plus four sine/cosine frequency pairs: 9 channels |
| Energy cues | Detached channel-mean absolute activation, spatially normalized and clipped: 2 channels |
| Gate | 11 → 32 → 2 pointwise convolutions with ReLU, then modality softmax |
| Projections | Separate bias-free linear 1 × 1 camera/LiDAR convolutions |
| Residual | Concatenated weighted projections → bias-free 3 × 3 convolution |
| Output | ReLU(reference + residual) |
| Initialization | Zero residual output kernel; zero final gate kernel and bias |

The weights describe feature allocation **inside the residual branch**. They are not calibrated sensor confidence and do not measure a modality's complete contribution to the detector. “Baseline-preserving” describes equality at initialization; it does not freeze the reference or promise that later outputs remain unchanged.

See [the method specification](docs/method.md) for equations, coordinate conventions, and the staged gradient flow caused by the two zero-initialized layers.

## Repository layout

```text
src/bpdaf/              Standalone PyTorch fusion package
tests/                 Numerical and behavior tests
examples/              Synthetic usage and training example
integrations/          Optional detector adapter
docs/method.md         Equations and implementation contract
docs/integration.md    Detector and checkpoint integration
docs/reproduction.md   Experimental settings and missing artifacts
.github/workflows/     CPU continuous integration
CITATION.cff           Citation metadata without an invented publication DOI
```

## Development

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
```

For contributions, follow [CONTRIBUTING.md](CONTRIBUTING.md). Report benchmark numbers only with the matching configuration, checkpoint, evaluation protocol, and logs.

## Citation and license

The associated manuscript is **“Baseline-Preserving Distance-Aware Adaptive Fusion for LiDAR-Camera 3D Object Detection”**, by Jiaxin Yang, Yongbiao Li, Zhanlin Cao, Long Chen, and Jinglong Wang. Publication details will be added when confirmed; see [CITATION.cff](CITATION.cff). Source repository: [3079Code/BPDAF](https://github.com/3079Code/BPDAF).

This implementation is provided under the [MIT License](LICENSE). Dataset access and any external detector code remain subject to their own terms.
