"""Tools for image processing and analysis.

This module contains tools for image processing and analysis.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from radimgarray import RadImgArray, SegImageArray


def zero_pad_to_square(array, k_space: bool = False):
    """Zero pad an array to make it square.
    !This is image space zero padding!
    Args:
        array (np.ndarray): The array to pad.
        k_space (bool): If True, the zero padding is done in k-space.
    Returns:
        np.ndarray: The zero padded array.
    """

    if array.shape[0] == array.shape[1]:
        return None

    if not k_space:
        new_array = pad_array(array, get_pad(array))
    else:
        if np.iscomplexobj(array):
            fft_array = np.fft.fftn(array)
            fft_array = np.fft.fftshift(fft_array)
            new_fft_array = pad_array(fft_array, get_pad(fft_array))
            new_array = np.fft.ifftn(new_fft_array)
        else:
            return None
    return new_array


def get_pad(array: np.ndarray) -> np.ndarray | None:
    diff = int(abs(array.shape[0] - array.shape[1]))
    if array.shape[0] < array.shape[1]:
        pad_size = [diff // 2, *list(array.shape[1:])]
    else:
        pad_size = [array.shape[0], diff // 2, *list(array.shape[2:])]
    return np.zeros(pad_size)


def pad_array(array: np.ndarray, pad: np.ndarray):
    return (
        np.vstack((pad, array, pad))
        if array.shape[0] < array.shape[1]
        else np.hstack((pad, array, pad))
    )


def get_mean_signal(
    img: RadImgArray | np.ndarray, seg: SegImageArray, value: int
) -> np.ndarray:
    """Get the mean signal of a specific segmentation.

    Args:
        img (RadImgArray, np.ndarray): The image data.
        seg (SegImageArray): The segmentation data
        value (int): The segmentation value to find.
    Returns:
        (list): The mean signal of the segmentation value.
    Raises:
        ValueError: If the segmentation value is not found in the array.
    """

    if img.shape[:-1] != seg.shape[:-1]:
        raise ValueError("Image and segmentation shape do not match")

    if value in seg.seg_values:
        array = np.where(seg == value)
        img[not array] = np.nan
        return np.nanmean(img, axis=(0, 1, 2))
    else:
        raise ValueError(f"Segmentation value {value} not found in array")


def get_single_seg_array(seg: SegImageArray, value: int) -> SegImageArray:
    """Get a new array with only a single segmentation presented by value.

    Args:
        seg (SegImageArray): The segmentation array.
        value (int): The segmentation value to extract.
    Returns:
        (SegImageArray): The new array with only the segmentation value.
    """
    if value in seg.seg_values:
        new_array = np.where(seg == value, 1, 0)
        return SegImageArray(new_array, info=seg.info)
    else:
        raise ValueError(f"Segmentation value {value} not found in array")


def save_mean_seg_signals_to_excel(
    img: RadImgArray, seg: SegImageArray, b_values: np.ndarray, path: Path
):
    """Save the mean segmentation signal to an Excel file.

    Args:
        img (RadImgArray): The image data.
        seg (SegImageArray): The segmentation data.
        path (Path): The path to save the Excel file.
    See Also:
        :func:`get_mean_signal`
    """
    _dict = {"index": b_values.squeeze().tolist()}
    for value in seg.seg_values:
        mean_signal = get_mean_signal(img, seg, value)
        _dict[value] = mean_signal.tolist()
    df = pd.DataFrame(_dict).T
    df.to_excel(path, header=False)


def array_to_rgba(array: np.ndarray) -> np.ndarray:
    """Convert an array to RGBA format.

    Args:
        array (np.ndarray): The array to convert.
    Returns:
        (np.ndarray): The array in RGBA format.
    """
    if array.ndim == 2:
        return np.repeat(array[:, :, np.newaxis], 4, axis=2)
    elif array.ndim == 3:
        return np.repeat(array, 4, axis=2)
    else:
        raise ValueError("Array must be 2D or 3D")


def slice_to_rgba(
    img: RadImgArray | np.ndarray, slice_num: int, alpha: int = 1
) -> np.ndarray:
    """Convert a 2D, 3D or 4D array slice to RGBA format.

    For 4D arrays, the first channel is selected.s
    Args:
        img (RadImgArray, np.ndarray): The image array to convert.
        slice_num (int): The slice number to convert.
        alpha (int): The alpha value of the RGBA format.
    Returns:
        (np.ndarray): The slice in RGBA format.
    """
    if img.ndim == 4:
        img = img[:, :, slice_num, 0]
    elif img.ndim == 3:
        img = img[:, :, slice_num]
    elif img.ndim == 2:
        img = img
    else:
        raise ValueError("Array must be 2D or 3D or 4D")

    array = np.rot90(img)
    if not np.nanmax(array) == np.nanmin(array):
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    else:
        array_norm = array / np.nanmax(array)
    alpha_map = np.full(array_norm.shape, alpha)
    return np.dstack((array_norm, array_norm, array_norm, alpha_map))
