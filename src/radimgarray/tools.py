from __future__ import annotations

import numpy as np

from radimgarray import RadImgArray, SegImageArray


def zero_pad_to_square(array, k_space: bool = False):
    """
    Zero pad an array to make it square.
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
            fft_array = np.fft.rfftn(array)
            new_fft_array = pad_array(fft_array, get_pad(fft_array))
            new_array = np.fft.irfftn(new_fft_array)

    if not isinstance(array, np.ndarray):
        return type(array)(new_array)
    else:
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
