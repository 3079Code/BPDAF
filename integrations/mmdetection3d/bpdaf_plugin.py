"""MMDetection3D registry adapter, independently written for BPDAF.

Target: open-mmlab/mmdetection3d at
fe25f7a51d36e3702f961e198894580d83c4387b, projects/BEVFusion.
The target has been inspected statically; full detector training and evaluation
have not been run for this new implementation.
"""

from collections.abc import Sequence

from mmdet3d.registry import MODELS

from bpdaf import BPDAFFuser


@MODELS.register_module(name="BPDAFFuser")
class MMDetection3DBPDAF(BPDAFFuser):
    """Use BPDAF at the pinned BEVFusion ``fusion_layer`` interface.

    Args:
        bev_range: Physical bounds ``(x_min, y_min, x_max, y_max)`` in meters.
        **kwargs: Remaining :class:`bpdaf.BPDAFFuser` constructor options.

    The host supplies ``[camera, lidar]`` in ``[B, C, X, Y]`` layout. The
    standalone core labels its last two axes ``[Y, X]``. Its geometry uses only
    radial distance, so relabeling the bounds gives the correct physical radius
    at every host cell without transposing tensors or reference convolution
    kernels. The host's numerical ConvFuser state-dict keys are retained.
    """

    def __init__(
        self,
        bev_range: Sequence[float] = (-54.0, -54.0, 54.0, 54.0),
        **kwargs,
    ) -> None:
        if len(bev_range) != 4:
            raise ValueError("bev_range must contain x_min, y_min, x_max, y_max")
        x_min, y_min, x_max, y_max = map(float, bev_range)
        super().__init__(
            bev_range=(y_min, x_min, y_max, x_max),
            **kwargs,
        )
        self.physical_bev_range = (x_min, y_min, x_max, y_max)
