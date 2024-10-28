import numpy as np

from radimgarray import RadImgArray


def behaves_like_ndarray(rad_img: RadImgArray, np_array: np.ndarray):
    # Check if shapes are the same
    assert rad_img.shape == np_array.shape

    # Check if all elements are the same
    assert np.allclose(rad_img, np_array)

    # Check if max values are the same
    assert rad_img.max() == np_array.max()

    # Check if sum values are the same
    assert rad_img.sum() == np_array.sum()

    # Check if mean values are the same
    assert rad_img.mean() == np_array.mean()

    # Check if slicing works the same way
    assert np.allclose(rad_img[1], np_array[1])
    assert np.allclose(rad_img[:, 1], np_array[:, 1])
    assert np.allclose(rad_img[1, :, 1], np_array[1, :, 1])

    # Check if reshaping works the same way
    assert np.allclose(
        rad_img.reshape(
            rad_img.shape[0] * rad_img.shape[-1], rad_img.shape[1], rad_img.shape[2]
        ),
        np_array.reshape(
            np_array.shape[0] * np_array.shape[-1], np_array.shape[1], np_array.shape[2]
        ),
    )

    # Check if transposing works the same way
    assert np.allclose(rad_img.transpose(), np_array.transpose())


def test_empty_assignment():
    try:
        RadImgArray([])
    except TypeError:
        assert True


def test_from_array():
    np_array = np.random.rand(3, 3, 3, 3)
    rad_img = RadImgArray(np_array)
    behaves_like_ndarray(rad_img, np_array)
