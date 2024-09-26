from radimgarray import RadImgArray
import numpy as np
from pathlib import Path

def test_from_nii(nifti_file):
    rad_img = RadImgArray()
    rad_img.load(nifti_file)
    assert rad_img.shape == (178, 178, 3, 0)

