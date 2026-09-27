"""BPDAF integration example; inherits the upstream six-epoch fusion recipe.

This is not the recovered configuration for the manuscript's reported runs.
Install this directory as ``projects/BPDAF`` in the pinned MMDetection3D tree.
"""

_base_ = [
    "../../BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
]

custom_imports = dict(
    imports=["projects.BEVFusion.bevfusion", "projects.BPDAF"],
    allow_failed_imports=False,
)

model = dict(
    fusion_layer=dict(
        _delete_=True,
        type="BPDAFFuser",
        in_channels=[80, 256],
        out_channels=256,
        # Physical (x_min, y_min, x_max, y_max); adapter handles host axis order.
        bev_range=[-54.0, -54.0, 54.0, 54.0],
        num_frequencies=4,
        gate_hidden_channels=32,
        energy_eps=1e-6,
        energy_clip=5.0,
        include_distance=True,
        include_energy=True,
        detach_energy=True,
        gate_mode="softmax",
        zero_init_residual=True,
    )
)
