"""Tools for image processing and analysis.

This module contains tools for image processing and analysis.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from turtle import st

import numpy as np
import pandas as pd

from .base_image import ImgArray
from .seg_image import SegArray


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
    img: ImgArray | np.ndarray, seg: SegArray, value: int
) -> np.ndarray:
    """Get the mean signal of a specific segmentation.

    Args:
        img (ImgArray, np.ndarray): The image data.
        seg (SegArray): The segmentation data
        value (int): The segmentation value to find.
    Returns:
        (list): The mean signal of the segmentation value.
    Raises:
        ValueError: If the segmentation value is not found in the array.
    """
    if img.ndim == 4:
        if seg.ndim == 4:
            if img.shape[:-1] != seg.shape[:-1]:
                raise ValueError("Image and segmentation shape do not match")
        elif seg.ndim == 3:
            if img.shape[:-1] != seg.shape:
                raise ValueError("Image and segmentation shape do not match")
    elif img.ndim == 3:
        if img.shape != seg.shape:
            raise ValueError("Image and segmentation shape do not match")
    elif img.ndim <= 2:
        raise ValueError("Image must be 3D or 4D")

    if seg.ndim <= 2 or seg.ndim > 4:
        raise ValueError("Segmentation must be 3D or 4D")

    if value in seg.seg_values:
        mask = seg == value
        if seg.ndim == 4:
            if seg.shape[-1] != 1:
                warnings.warn(
                    "Segmentation array has multiple channels, only the first channel will be used",
                    UserWarning,
                    stacklevel=2,
                )
                mask = mask[..., 0]
            if mask.ndim == 4:
                mask = mask.squeeze(axis=3)
            img_masked = img[mask]
        else:
            img_masked = img[mask]
        return np.mean(img_masked, axis=0)
    else:
        raise ValueError(f"Segmentation value {value} not found in array")


def get_single_seg_array(seg: SegArray, value: int) -> SegArray:
    """Get a new array with only a single segmentation presented by value.

    Args:
        seg (SegImageArray): The segmentation array.
        value (int): The segmentation value to extract.
    Returns:
        (SegImageArray): The new array with only the segmentation value.
    """
    if value in seg.seg_values:
        new_array = np.where(seg == value, 1, 0)
        return SegArray(new_array, info=seg.info)
    else:
        raise ValueError(f"Segmentation value {value} not found in array")


def save_mean_seg_signals_to_excel(
    img: ImgArray, seg: SegArray, b_values: np.ndarray, path: Path
):
    """Save the mean segmentation signal to an Excel file.

    Args:
        img (ImgArray): The image data.
        seg (SegArray): The segmentation data.
        b_values (np.ndarray): The b-values of the image.
        path (Path): The path to save the Excel file.
    See Also:
        :func:`get_mean_signal`
    """
    _dict = {
        "index": (
            b_values.squeeze().tolist()
            if isinstance(b_values, np.ndarray)
            else b_values
        )
    }
    for value in seg.seg_values:
        mean_signal = get_mean_signal(img, seg, value)
        _dict[value] = mean_signal.tolist()
    df = pd.DataFrame(_dict).T
    df.to_excel(path, header=False)


def array_to_rgba(
    array: np.ndarray, alpha: float | np.float | np.ndarray = 1.0
) -> np.ndarray:
    """Convert an array to RGBA format.

    2D, 3D and 4D arrays are supported. The RGBA dimension will always be the third in
    the array. The final shape of the new array will similar to:
    (x, y, rgba, z, t) for (x, y, z, t) array.

    Args:
        array (np.ndarray): The array to convert.
        alpha (float, np.ndarray):
    Returns:
        (np.ndarray): The array in RGBA format.
    """
    if isinstance(alpha, np.ndarray):
        if not alpha.shape == array.shape:
            raise ValueError("Alpha array must have the same shape as the input array")
        alpha_array = alpha
    else:
        alpha_array = None
    if array.ndim == 2:
        new_array = array[:, :, np.newaxis]
        if alpha_array is None:
            alpha_array = np.full(new_array.shape, alpha)
        new_array = np.repeat(new_array, 3, axis=2)
        return np.dstack((new_array, alpha_array))
    elif array.ndim == 3:
        new_array = array[:, :, :, np.newaxis]
        if alpha_array is None:
            alpha_array = np.full(new_array.shape, alpha)
        new_array = np.repeat(new_array, 3, axis=3)
        return np.transpose(
            np.concatenate((new_array, alpha_array), axis=-1), (0, 1, 3, 2)
        )
    elif array.ndim == 4:
        new_array = array[:, :, :, :, np.newaxis]
        if alpha_array is None:
            alpha_array = np.full(new_array.shape, alpha)
        new_array = np.repeat(new_array, 3, axis=4)
        return np.transpose(
            np.concatenate((new_array, alpha_array), axis=-1), (0, 1, 4, 2, 3)
        )
    else:
        raise ValueError("Array must be 2D or 3D")


def slice_to_rgba(
    img: ImgArray | np.ndarray, slice_num: int, alpha: float | np.float = 1
) -> np.ndarray:
    """Convert a 2D, 3D or 4D array slice to RGBA format.

    For 4D arrays, the first channel is selected.s
    Args:
        img (RadImgArray, np.ndarray): The image array to convert.
        slice_num (int): The slice number to convert.
        alpha (float): The alpha value of the RGBA format.
    Returns:
        (np.ndarray): The slice in RGBA format.
    """
    if not alpha <= 1 and not alpha > 0:
        raise ValueError("Alpha must be between 0 and 1")
    if img.ndim == 4:
        img = img[:, :, slice_num, 0]
    elif img.ndim == 3:
        img = img[:, :, slice_num]
    elif img.ndim == 2:
        img = img
    else:
        raise ValueError("Array must be 2D, 3D or 4D")

    array = np.rot90(img)
    alpha_map = np.full(array.shape, alpha)[:, :, np.newaxis]
    if not np.nanmax(array) == np.nanmin(array):
        array_norm = (
            (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        )[:, :, np.newaxis]
        array_norm = np.repeat(
            array_norm,
            3,
            axis=2,
        )
    else:
        array_norm = np.repeat((array / np.nanmax(array)), 3, axis=2)
    return np.concatenate((array_norm, alpha_map), axis=2)
