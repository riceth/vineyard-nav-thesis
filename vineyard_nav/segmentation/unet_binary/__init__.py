"""Phase A — U-Net binary baseline (SMP + ImageNet ResNet-34).

See docs/PHASE_A_SPEC.md for the implementation contract.
"""

from .dataset import (
    SemanticBLTBinaryDataset,
    canopy_state_from_filename,
    train_transform,
    eval_transform,
    FOREGROUND_CATEGORY_IDS,
)
from .model import UNetBinary, build_model

__all__ = [
    "SemanticBLTBinaryDataset",
    "canopy_state_from_filename",
    "train_transform",
    "eval_transform",
    "FOREGROUND_CATEGORY_IDS",
    "UNetBinary",
    "build_model",
]
