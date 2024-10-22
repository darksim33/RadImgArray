from __future__ import annotations

import numpy as np

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


def get_mean_signal(
    img: RadImgArray | np.ndarray, seg: SegImageArray, value: int
) -> list:
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
        return [np.nanmean(img, axis=3)]
    else:
        raise ValueError(f"Segmentation value {value} not found in array")


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
