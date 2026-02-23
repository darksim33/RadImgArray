from __future__ import annotations

from pathlib import Path

import numpy as np

from .base_image import ImgArray


class SegArray(ImgArray):
    """A class for storing and manipulating segmentation images.

    Attributes:
        seg_values (np.ndarray): Unique segmentation values in the array.
        number_segs (int): Number of unique segmentation values in the array.
    """

    seg_values: list[int]
    number_segs: int

    def __new__(cls, _input: np.ndarray | list | Path | str, *args, **kwargs):
        """Create a new instance of SegImageArray.

        Args:
            _input: (np.ndarray, list, Path, str): Input data for the array.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            SegImageArray: A new instance of SegImageArray.
        """
        obj = super().__new__(cls, _input, *args, **kwargs)
        obj.seg_values = cls.__get_seg_values(obj)
        obj.number_segs = len(obj.seg_values)
        return obj

    def __array_finalize__(self, obj):
        # NOTE: the array_finalize helps keep the metadata when creating new arrays or
        # when performing operations on arrays
        super().__array_finalize__(obj)
        self.seg_values = getattr(obj, "seg_values", [])
        self.number_segs = getattr(obj, "number_segs", 0)

    @classmethod
    def __get_seg_values(cls, obj) -> list[int]:
        """Get the unique segmentation values in the array.

        Returns:
            list[int]: Unique segmentation values in the array.
        """
        seg_values = np.unique(obj).astype(int).tolist()
        if 0 in seg_values:
            seg_values.remove(0)  # excluding zero since it's not a segmentation
        return seg_values

    def get_seg_indices(self, value: int) -> list[tuple]:
        """Get the indices of a specific segmentation value in the array.

        Args:
            value (int): The segmentation value to find.
        Returns:
            list(tuple): Indices where the segmentation value is found.
        Raises:
            ValueError: If the segmentation value is not found in the array.
        """
        if value in self.seg_values:
            raw_indices = np.where(self == value)
            return list(zip(*raw_indices))
        else:
            raise ValueError(f"Segmentation value {value} not found in array")

    def zero_pad(self):
        pass

    # def get_single_seg_array(self, value: int):
    #     """
    #     Get a new array with only a single segmentation presented by value.
    #     Args:
    #         value: int: The segmentation value to extract.
    #     Returns:
    #         SegImageArray: A new array with only the specified segmentation value.
    #     """
    #     if value not in self.seg_values:
    #         raise ValueError(f"Segmentation value {value} not found in array")
    #     else:
    #         return SegImageArray(self == value)
