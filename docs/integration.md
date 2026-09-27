# OpenMMLab MMDetection3D integration

The optional adapter targets the official OpenMMLab `projects/BEVFusion`
implementation at commit
`fe25f7a51d36e3702f961e198894580d83c4387b`. It uses MMEngine's native config
inheritance, `custom_imports`, model registry and training/testing entrypoints.
It is separate from the archived MIT-HAN-Lab fork, which uses another registry,
Torchpack YAML files and a different detector interface.

**Validation status:** the host interfaces, initialization path, feature order
and spatial indexing were checked by static inspection of that official commit.
Full MMDetection3D/CUDA installation, detector training and dataset evaluation
have not been run for this new implementation. The example inherits the
upstream six-epoch fusion schedule; it is not the manuscript's recovered
24-epoch training configuration or a reproduction of its reported metrics.

## Environment

The manuscript reports Python 3.10, PyTorch 2.1.2 and MMEngine 0.10.7. The selected
modern integration uses the corresponding API generation. At the pinned
MMDetection3D commit, package guards accept MMEngine `>=0.8.0,<1.0.0`, MMCV
`>=2.0.0rc4,<2.2.0` and MMDetection `>=3.0.0rc5,<3.4.0`. An environment using
MMEngine 0.10.7, MMCV 2.1.0 and MMDetection 3.3.0 falls within these guards.
This version check alone does not establish binary compatibility: build/install
MMCV and the BEVFusion CUDA extensions for the actual PyTorch and CUDA versions
on the training machine. Follow the pinned upstream installation documentation.
Do not mix `mmcv-full` 1.x or the MIT fork into this environment.

The integration does not vendor OpenMMLab source, dataset files or pretrained
weights. Their original licenses and download conditions remain applicable.

## Install into a dedicated upstream checkout

The following commands are Linux shell examples. Replace `/path/to/bpdaf-public`
with the absolute path to this repository, and run in an environment with the
required PyTorch/CUDA/MMCV dependencies already installed.

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout fe25f7a51d36e3702f961e198894580d83c4387b
python -m pip install -v -e .
python projects/BEVFusion/setup.py develop
python -m pip install -e /path/to/bpdaf-public

# Copy only when the target does not already exist.
test ! -e projects/BPDAF && \
  cp -R /path/to/bpdaf-public/integrations/mmdetection3d projects/BPDAF
```

The copied layout is intentional: `projects/BPDAF/configs/bpdaf_nuscenes.py`
inherits a sibling `projects/BEVFusion/configs/...` file. Running the example
directly from the BPDAF repository does not supply that upstream config tree.
The copy runs only if the target is absent, avoiding a nested second copy in an
existing project directory.

Prepare nuScenes using this pinned MMDetection3D version's data preparation
instructions and keep its expected `data/nuscenes` layout. The MIT fork's old
metadata format is not interchangeable with current MMDetection3D metadata.

## Check configuration and registration

From the MMDetection3D checkout root:

```bash
python -c "from mmengine.config import Config; from mmdet3d.registry import MODELS; c=Config.fromfile('projects/BPDAF/configs/bpdaf_nuscenes.py'); f=MODELS.build(c.model.fusion_layer); print(type(f).__name__); print(c.train_cfg); print(f.physical_bev_range)"
```

Expected registration is `MMDetection3DBPDAF`; physical bounds are
`(-54.0, -54.0, 54.0, 54.0)`. This command constructs the fuser on CPU. The host
runner subsequently moves the detector to the selected device. The adapter
does not allocate a GPU or choose a device in its constructor.

Before a long run, construct the complete model in the target environment and
run a real mini-batch in both training and inference mode. Check the two tensors
entering `fusion_layer`: camera first, LiDAR second, matching batch size and
spatial dimensions, with channel counts 80 and 256 respectively.

## Train and evaluate with native host commands

Set `LIDAR_CHECKPOINT` to a compatible OpenMMLab LiDAR-only checkpoint and
`IMAGE_CHECKPOINT` to the documented Swin image-backbone checkpoint. Acquire
these separately from the official upstream links. They are not BPDAF weights.

```bash
export LIDAR_CHECKPOINT=/path/to/lidar-only-checkpoint.pth
export IMAGE_CHECKPOINT=/path/to/swint-nuimages-pretrained.pth

bash tools/dist_train.sh projects/BPDAF/configs/bpdaf_nuscenes.py 8 \
  --work-dir work_dirs/bpdaf_example \
  --cfg-options load_from="$LIDAR_CHECKPOINT" \
    model.img_backbone.init_cfg.checkpoint="$IMAGE_CHECKPOINT"
```

For a matched ConvFuser run, substitute
`projects/BPDAF/configs/convfuser_nuscenes.py` and another work directory. Apply
identical data, seed, batch-size, optimizer, schedule, augmentation and
initialization choices to both runs. To use AMP, add the host's `--amp` option
consistently to both commands. Do not change only `max_epochs` to 24: the learning
rate/momentum schedules, warmup and checkpoint policy must be specified together
before calling that a manuscript reproduction configuration.

Evaluate the newly trained BPDAF checkpoint with the same BPDAF config:

```bash
bash tools/dist_test.sh projects/BPDAF/configs/bpdaf_nuscenes.py \
  work_dirs/bpdaf_example/epoch_6.pth 8
```

The inherited test dataset/evaluator uses the **nuScenes validation split**.
The filename `test.py` does not make this a nuScenes challenge-test evaluation.
Official test-set submission requires the appropriate dataset/config and
prediction-formatting workflow from upstream; do not report validation output
as test-server results. For single-GPU operation use the upstream
`python tools/train.py CONFIG ...` / `python tools/test.py CONFIG CHECKPOINT ...`
entrypoints and explicitly record any changed effective batch size.

## Initialization and checkpoint behavior

The adapter registers the core as `BPDAFFuser` in `mmdet3d.registry.MODELS` and
occupies `model.fusion_layer`. The unchanged ConvFuser reference keeps its
numerical module keys: `fusion_layer.0.weight` for the convolution and
`fusion_layer.1.*` for BatchNorm parameters and running statistics. BPDAF adds
`fusion_layer.adaptive.*` parameters. A same-architecture ConvFuser checkpoint
can therefore initialize the reference without renaming those keys.

Loading a baseline with the host's `load_from` necessarily leaves adaptive
parameters absent from that checkpoint. Inspect the loading log: for an otherwise
identical full fusion detector, missing keys should be limited to the new
adaptive branch. Starting from the documented **LiDAR-only** checkpoint also
has expected missing image and fusion parameters, and does not transfer a
trained ConvFuser reference. These are different initialization protocols and
must be recorded distinctly.

The pinned BEVFusion constructor calls its `init_weights()` method, whose
implementation initializes only the image backbone. It does not reinitialize
the fusion layer. Thus static inspection finds no host step that overwrites the
core's zero-output residual initialization. This must still be checked after
checkpoint loading in the actual environment; resumed BPDAF checkpoints are
expected to contain learned, nonzero residual parameters. Do not reset those
parameters when resuming training or evaluating a trained model.

Functional preservation means equality with the reference fusion function at
BPDAF initialization using the same reference weights, BN state and mode. The
reference remains trainable. It does not imply permanent equality during
optimization, or unchanged accuracy merely from selecting a compatible
checkpoint format.

## Spatial axes and grid alignment

The standalone core interprets tensor storage as `[B,C,Y,X]`. The pinned
OpenMMLab BEVFusion implementation stores fused inputs as `[B,C,X,Y]`:

1. Camera pooling passes the x-bin count as its first spatial dimension and the
   y-bin count as its second; the CUDA address calculation indexes x before y.
2. Its project-specific voxelizer emits x, y, z indices. The corresponding
   sparse encoder preserves the first two dimensions when flattening z into
   channels. A generic sparse-encoder docstring mentions another order; the
   actual project-specific indexing determines this adapter's behavior.
3. The detector appends camera features before LiDAR features.

The adapter accepts ordinary physical bounds `(x_min,y_min,x_max,y_max)` and
passes `(y_min,x_min,y_max,x_max)` into the core. This relabels the core's radial
grid so that every radius corresponds to the host's actual cell. It does not
transpose either feature tensor or convolution kernel, preserving reference
checkpoint behavior. Because the current geometry uses only radius, exchanging
the coordinate labels preserves `sqrt(x*x+y*y)`. A future directional/angle
descriptor would require its own axis-aware adapter.

For the supplied inherited nuScenes config, x/y extent is 108 m. Camera pooling
uses 0.3 m bins followed by downsampling by two: `360/2=180` cells per axis.
LiDAR uses 0.075 m voxels and an output factor of eight: `1440/8=180` cells.
Both branches therefore reach the fuser at `180 x 180`; output channels remain
256. This is a configuration consistency check, not an end-to-end calibration
validation.

If ranges or resolution change, update the camera `xbound/ybound`, voxelization
range and size, sparse shape, point/object range filters, detection-head grid
sizes/coder ranges, and BPDAF `bev_range` consistently. Passing a rectangular
range into only the fuser cannot repair unaligned input features. This example
does not claim to supply the manuscript's KITTI detector/data pipeline.

## Official sources inspected

All MMDetection3D links below are pinned to the integration target.

- [BEVFusion project and host commands](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/README.md)
- [Host detector, feature order and initialization](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/bevfusion/bevfusion.py)
- [ConvFuser implementation](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/bevfusion/transfusion_head.py)
- [Camera and LiDAR fusion configuration](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py)
- [LiDAR configuration and grid parameters](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py)
- [Camera BEV pooling dimensions](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/bevfusion/depth_lss.py)
- [BEV pooling CUDA indexing](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/bevfusion/ops/bev_pool/src/bev_pool_cuda.cu)
- [Project-specific voxel CUDA indexing](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/bevfusion/ops/voxel/src/voxelization_cuda.cu)
- [Project-specific sparse encoder](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion/bevfusion/sparse_encoder.py)
- [Dependency version guards](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/__init__.py)
- [Official installation guidance](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/docs/en/get_started.md)
