"""Data loaders for StageBridge.

Provides Dataset and DataLoader classes for receiver-centered niche data
with individual cells per ring (for learned ISAB+PMA pooling).
"""

from stagebridge.loaders.dataset import (
    StageBridgeDataset,
    NicheBatch,
    collate_niche_batch,
    create_dataloaders,
)
from stagebridge.loaders.splits import (
    load_split_manifest,
    get_fold_donors,
    SplitManifest,
)

__all__ = [
    "StageBridgeDataset",
    "NicheBatch",
    "collate_niche_batch",
    "create_dataloaders",
    "load_split_manifest",
    "get_fold_donors",
    "SplitManifest",
]
