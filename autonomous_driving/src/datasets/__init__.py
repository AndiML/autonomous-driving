"""A sub-package that contains datasets for 2D object detection."""

from autonomous_driving.src.datasets.kitti import KITTI

DATASET_IDS = [
    KITTI.dataset_id,
]
"""Contains the IDs of all available datasets."""

DEFAULT_DATASET_ID = KITTI.dataset_id
"""Contains the ID of the default dataset."""

__all__ = [
    'Kitti'

    'DATASET_IDS',
    'DEFAULT_DATASET_ID'
]
