"""DA-Fusion: Deformable Attention-Based RGB-D Fusion Transformer (ICRA 2025).

Importing this package registers the config, datasets, dataset mapper, backbone,
and meta-architecture with detectron2.
"""
from .config import add_dafusion_config  # noqa: F401
from . import modeling  # noqa: F401  (registers DAFusion arch, dual backbone, heads)
from . import data      # noqa: F401  (registers datasets + mapper)
