from __future__ import annotations
import numpy as np
from pathlib import Path

from .base_image import RadImgArray

class SegImageArray(RadImgArray):
    def __new__(cls, _input: np.ndarray | list | Path | str, *args, **kwargs):
        obj = super().__new__(cls, _input, *args, **kwargs)
        return obj

    def __init__(self, _input: np.ndarray | list | Path | str, *args, **kwargs):
        super().__init__(_input, *args, **kwargs)

        self.segmentation_values = np.unique(self).astype(int)
        self.number_segmentations = int(np.unique(self).shape[0])

    def __array_finalize__(self, obj, /):
        super().__array_finalize__(obj)

    def get_segmentation_indices(self, value: int):
        return np.where(self == value)

