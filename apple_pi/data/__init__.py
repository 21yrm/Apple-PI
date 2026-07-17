"""Ground-truth and prediction data contracts."""

from .case import CaseData
from .dataset import DatasetIndex, load_dataset_index
from .validation import validate_ground_truth, validate_predictions

__all__ = [
    "CaseData",
    "DatasetIndex",
    "load_dataset_index",
    "validate_ground_truth",
    "validate_predictions",
]
