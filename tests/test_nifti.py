from radimgarray import RadImgArray
import numpy as np
import pytest


# def test_rad_img_array():
test = np.random.rand(3, 3, 3, 3)
rad_img = RadImgArray(test)
rad_img.show()
