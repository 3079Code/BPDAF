"""Optional adapter for the pinned OpenMMLab MMDetection3D BEVFusion project.

Copy this directory to ``mmdetection3d/projects/BPDAF``; see
``docs/integration.md`` in the BPDAF repository.
"""

from .bpdaf_plugin import MMDetection3DBPDAF

__all__ = ["MMDetection3DBPDAF"]
