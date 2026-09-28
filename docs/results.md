# Reported experimental results

This page records the final numerical results reported in the BPDAF manuscript.
It is a compact result summary rather than a raw training-log archive. Dataset
splits, optimization settings, software versions, hardware, and the boundary of
the released artifacts are documented in [reproduction.md](reproduction.md).

Unless noted otherwise, mAP, NDS, AP, and their differences are percentage
values or percentage-point changes. ConvFuser and BPDAF rows are matched local
runs. The DepthFusion row is a published external reference and was not
reproduced in this work.

## nuScenes ablation study

Manuscript Table 3: BPDAF components on the nuScenes validation set.

| Variant | Reference path | Radial code | Energy cues | Stop-gradient | Softmax gate | Zero initialization | mAP (%) | NDS (%) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| ConvFuser baseline | Yes | No | No | No | No | No | 63.70 | 67.26 |
| No distance code | Yes | No | Yes | Yes | Yes | Yes | 64.22 | 67.88 |
| No energy cues | Yes | Yes | No | Yes | Yes | Yes | 63.87 | 67.40 |
| No stop-gradient | Yes | Yes | Yes | No | Yes | Yes | 64.61 | 67.74 |
| Independent sigmoid gates | Yes | Yes | Yes | Yes | No | Yes | 64.17 | 67.59 |
| No baseline-preserving initialization | Yes | Yes | Yes | Yes | Yes | No | 64.39 | 67.95 |
| Vanilla adaptive gate (replacement) | No | Yes | Yes | Yes | Yes | No | 63.15 | 66.72 |
| **BPDAF (full)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **64.91** | **68.12** |

## nuScenes validation checkpoints

Manuscript Table 4: matched validation results under the same training budget.

| Epoch | ConvFuser mAP | BPDAF mAP | Delta mAP | ConvFuser NDS | BPDAF NDS | Delta NDS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 54.85 | 56.21 | +1.36 | 58.73 | 59.78 | +1.05 |
| 12 | 58.73 | 60.56 | +1.83 | 62.77 | 64.23 | +1.46 |
| 18 | 61.97 | 63.52 | +1.55 | 66.25 | 67.37 | +1.12 |
| 24 | 63.70 | 64.91 | +1.21 | 67.26 | 68.12 | +0.86 |

## nuScenes test-set evaluation

Manuscript Table 5: matched inference protocol.

| Method | mAP (%) | NDS (%) | mATE | mASE | mAOE | mAVE | mAAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvFuser baseline | 63.42 | 67.12 | 0.309 | 0.262 | 0.365 | 0.301 | 0.222 |
| **BPDAF** | **64.63** | **68.19** | **0.298** | **0.257** | **0.350** | **0.289** | **0.219** |

## Context against a published method

Manuscript Table 6. The DepthFusion-large values are taken from Table I of the
preprint version cited by the manuscript (arXiv:2505.07398v1). Its input
resolution, fusion pipeline, and training configuration differ from the matched
local runs, so this comparison does not isolate the fusion module.

| Method / source | Image backbone | Image size | Validation mAP / NDS | Test mAP / NDS |
| --- | --- | ---: | ---: | ---: |
| ConvFuser (this work) | Swin-T | 256 x 704 | 63.70 / 67.26 | 63.42 / 67.12 |
| BPDAF (this work) | Swin-T | 256 x 704 | 64.91 / 68.12 | 64.63 / 68.19 |
| DepthFusion-large (published) | Swin-T | 384 x 1056 | 72.3 / 74.4 | 72.8 / 75.4 |

## nuScenes with reduced LiDAR input

Manuscript Table 7: validation with 50% LiDAR point retention. Both models use
their epoch-24 checkpoints without retraining and receive the same retained-point
samples.

| LiDAR input | Method | mAP (%) | NDS (%) |
| --- | --- | ---: | ---: |
| Clean | ConvFuser | 63.70 | 67.26 |
| Clean | BPDAF | 64.91 | 68.12 |
| 50% retained | ConvFuser | 61.54 | 65.82 |
| 50% retained | **BPDAF** | **62.73** | **66.72** |

Under 50% retention, BPDAF remains 1.19 mAP and 0.90 NDS points above
ConvFuser.

## nuScenes class-wise validation AP

Manuscript Table 8: epoch-24 class-wise AP.

| Class | ConvFuser | BPDAF | Delta |
| --- | ---: | ---: | ---: |
| Car | 88.56 | 90.42 | +1.86 |
| Truck | 60.10 | 61.39 | +1.29 |
| Construction vehicle | 25.75 | 26.94 | +1.19 |
| Bus | 71.57 | 73.88 | +2.31 |
| Trailer | 41.45 | 41.18 | -0.27 |
| Barrier | 71.02 | 72.44 | +1.42 |
| Motorcycle | 70.90 | 71.34 | +0.44 |
| Bicycle | 57.53 | 59.15 | +1.62 |
| Pedestrian | 73.85 | 75.32 | +1.47 |
| Traffic cone | 76.28 | 77.04 | +0.76 |

The ten BPDAF entries average 64.91%, matching the aggregate epoch-24
validation mAP.

## KITTI validation

Manuscript Table 9: matched AP40 at the shared epoch-50 checkpoint, without
test-time augmentation.

| Entry | Easy | Moderate | Hard |
| --- | ---: | ---: | ---: |
| 2D ConvFuser | 80.06 | 71.88 | 67.79 |
| **2D BPDAF** | **81.84** | **72.07** | **69.15** |
| 2D delta | +1.78 | +0.19 | +1.36 |
| BEV ConvFuser | 74.67 | 64.07 | 60.60 |
| **BEV BPDAF** | **77.03** | **65.71** | **62.52** |
| BEV delta | +2.36 | +1.64 | +1.92 |
| 3D ConvFuser | 68.48 | 56.25 | 52.67 |
| **3D BPDAF** | **71.13** | **57.78** | **54.12** |
| 3D delta | +2.65 | +1.53 | +1.45 |

## Distance-stratified analysis

Manuscript Table 10: nuScenes validation objects grouped by radial distance.
Camera and LiDAR columns are mean residual-branch modality weights. Delta AP is
BPDAF AP minus ConvFuser AP on the same distance-restricted subset.

| Range | Ground-truth objects | Camera weight | LiDAR weight | Delta AP (pp) |
| --- | ---: | ---: | ---: | ---: |
| 0--20 m | 12,450 | 0.37 | 0.63 | +0.82 |
| 20--40 m | 8,320 | 0.46 | 0.54 | +1.57 |
| >40 m | 3,180 | 0.58 | 0.42 | +1.13 |

## Computational cost

Manuscript Table 11. Latency and peak memory were measured on an NVIDIA
A800-SXM4-80GB at batch size 1 with automatic mixed precision and identical
warm-up.

| Metric | ConvFuser | BPDAF | Increase |
| --- | ---: | ---: | ---: |
| Parameters (M) | 42.15 | 43.42 | +1.27 (3.0%) |
| FLOPs (G) | 86.30 | 127.32 | +41.02 (47.5%) |
| Latency (ms/frame) | 28.40 | 31.92 | +3.52 (12.4%) |
| Peak GPU memory (GB) | 12.70 | 13.58 | +0.88 (6.9%) |

## Artifact boundary

The tables above preserve the values and qualifications reported in the
manuscript. They do not replace raw training logs, evaluation exports,
checkpoints, or complete detector configurations, which are not included in
this public release. See [validation.md](validation.md) for package-level checks
and [reproduction.md](reproduction.md) for the complete artifact boundary.
