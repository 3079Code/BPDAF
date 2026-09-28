# Reproduction status

This repository implements the BPDAF method described in the manuscript. The repository maintainer reports completing nuScenes and KITTI reproduction with results matching the manuscript. This report is separate from the package's CPU tests and synthetic optimization checks. The public repository does not contain the original experimental source tree or the full reproduction artifacts listed below.

## Available in this release

| Artifact | Scope |
| --- | --- |
| Standalone BPDAF and ConvFuser | Forward computation and initialization behavior |
| Distance-energy context | Metric cell centers, radial Fourier features, relative modality activation |
| Tests | Module invariants, gradients, input handling, and state behavior |
| Synthetic example | Small forward/backward optimization run |
| Optional integration adapter | A starting point for a compatible external detector; see its documented limits |
| CPU CI definition | Package lint, tests, example, and build on two specified environments |
| Reported result tables | Final manuscript values collected in [results.md](results.md) |

## Settings recoverable from the manuscript

The following values describe the manuscript's experiments. Listing them does not turn this package into a complete reproduction configuration.

| Setting | nuScenes | KITTI |
| --- | --- | --- |
| Split | Official v1.0-trainval; 700 train and 150 validation scenes | 3,712 training / 3,769 validation samples |
| Inputs | Six cameras, current LiDAR frame plus nine historical sweeps | Front camera and LiDAR |
| Fusion channels | Camera 80, LiDAR 256, output 256 | Confirm in the complete detector configuration |
| BEV horizontal range | x,y ∈ [−54, 54] m | Exact numeric range absent from the supplied manuscript |
| Fusion grid | 180 × 180, cell width 0.6 m | Exact dimensions absent from the supplied manuscript |
| Radial frequencies / gate hidden channels | 4 / 32 | Same mechanism; verify exported config |
| Energy clipping / denominator floor | 5 / 1e−6 | Same mechanism; verify exported config |
| Training epochs / batch size | 24 / 8 | 60 / 4 |
| AdamW learning rate / weight decay | 4e−4 / 1e−2 | 2e−4 / 1e−2 |
| Schedule | 500-iteration linear warm-up, cosine decay to 2e−6 | Same warm-up and minimum learning rate |
| Precision / gradient clipping | AMP with dynamic scaling / max norm 10 | Same |
| Seed | 1788189656; deterministic=False | Same matched seed |
| Reported comparison checkpoints | Epochs 6, 12, 18, 24 | Shared epoch 50 |
| Reported device | NVIDIA A800-SXM4-80GB | NVIDIA RTX 4090 |

The manuscript reports Python 3.10.8, PyTorch 2.1.2, TorchVision 0.16.2, CUDA 11.8, cuDNN 8.7, MMEngine 0.10.7, and OpenCV 4.11.0. These values do not identify every detector dependency or its source revision.

## Reproduction artifacts not included in this release

| Artifact not included | Why it matters |
| --- | --- |
| Original detector repository and exact commit | BEVFusion implementations and checkpoint formats differ |
| Complete nuScenes and KITTI configurations | Encoder, head, losses, assignment, augmentations, class mapping, and sampling must match |
| Exact KITTI range and fusion-grid dimensions | They define the distance cue, not merely tensor shape |
| Pretrained weights and their provenance | Initialization affects the paired experiment |
| Split files and preprocessing metadata | Identical counts do not prove identical samples or processing |
| Original baseline/BPDAF checkpoints and logs | Needed to audit published scores and evaluation settings |
| Full ablation configurations | A named ablation alone does not specify every initialization and capacity control |
| Sparsification implementation and retained-point samples | Both detectors must receive exactly the same perturbation |
| Timing script, warm-up count, and FLOP-count convention | Needed to reproduce cost numbers on matched hardware |

Model checkpoints are not included in this release. Dataset files should be obtained from [nuScenes](https://www.nuscenes.org/) and [KITTI](https://www.cvlibs.net/datasets/kitti/) under their respective access terms.

## A defensible reproduction workflow

1. Record the chosen detector repository commit and install its required environment separately from the standalone package tests.
2. Export the full baseline configuration, dataset/split identifiers, pretrained-weight checksums, seed, and evaluation command.
3. Verify feature order, BEV axis convention, metric range, channel counts, normalization, and checkpoint compatibility at the fusion interface.
4. Copy the reference fuser, insert BPDAF, and verify initial fused-output equality on the same encoded features in the same mode.
5. Run a short end-to-end detector training/evaluation check before launching matched full runs. Keep all non-fusion settings fixed and record any unavoidable differences.
6. Preserve the full configurations, software versions, logs, evaluation outputs, and checkpoint checksums for both runs. Report new results as results of this reimplementation, with differences from the manuscript disclosed.

The original work uses one matched seed. A new repeated-seed study, capacity-matched control, or corruption evaluation should be identified as additional evidence, not attributed to the existing manuscript.
