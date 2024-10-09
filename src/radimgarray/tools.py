import numpy as np

from radimgarray import RadImgArray, SegImageArray


def zero_pad_to_square(array):
    """
    Zero pad an array to make it square.
    !This is image space zero padding!
    Args:
        array (np.ndarray): The array to pad.
    Returns:
        np.ndarray: The zero padded array.
    """
    if array.shape[0] == array.shape[1]:
        return array
    else:
        diff = int(abs(array.shape[0] - array.shape[1]))
        pad = (
            np.zeros((diff // 2, array.shape[1]))
            if array.shape[0] < array.shape[1]
            else np.zeros((array.shape[0], diff // 2))
        )
        new_array = (
            np.vstack((pad, array, pad))
            if array.shape[0] < array.shape[1]
            else np.hstack((pad, array, pad))
        )
        if not isinstance(array, np.ndarray):
            return type(array)(new_array)
        else:
            return new_array


def get_mean_signal(img: RadImgArray | np.ndarray, seg: SegImageArray, value: int):
    """
    Get the mean signal of a specific segmentation.
    Args:
        img: RadImgArray | np.ndarray: The image data.
        seg: SegImageArray: The segmentation data
        value: int: The segmentation value to find.
    Returns:
        float: The mean signal of the segmentation value.
    Raises:
        ValueError: If the segmentation value is not found in the array.
    """

    if img.shape[:-1] != seg.shape[:-1]:
        raise ValueError("Image and segmentation shape do not match")

    if value in seg.seg_values:
        array = np.where(seg == value)
        img[not array] = np.nan
        return np.nanmean(img, axis=3)
    else:
        raise ValueError(f"Segmentation value {value} not found in array")
