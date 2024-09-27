import numpy as np

from radimgarray import RadImgArray


def test_load_dicom(dicom_folder):
    rad_img = RadImgArray(dicom_folder)
    np_array = np.random.rand(176, 176, 15, 3)
    assert rad_img.shape == np_array.shape
