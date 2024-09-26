import numpy as np

from radimgarray import RadImgArray

def test_from_array():
    test = np.random.rand(3, 3, 3, 3)
    rad_img = RadImgArray(test)
    assert rad_img.shape == test.shape
    assert np.allclose(rad_img, test)

def test_empty():
    rad_img = RadImgArray()
    assert rad_img.shape == np.array([]).shape