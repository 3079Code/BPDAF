"""Matched upstream reference for the BPDAF integration example.

Use the same data, seed, optimizer, schedule and initialization choices for a
controlled comparison. This file itself does not reproduce manuscript results.
"""

_base_ = [
    "../../BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
]
