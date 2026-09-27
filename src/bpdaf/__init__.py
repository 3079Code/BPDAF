"""Paper-guided BPDAF fusion for aligned camera and LiDAR BEV features."""

from .context import DistanceEnergyContext
from .fusion import BPDAFFuser, ConvFuser

__all__ = ["BPDAFFuser", "ConvFuser", "DistanceEnergyContext"]
__version__ = "0.1.0"
