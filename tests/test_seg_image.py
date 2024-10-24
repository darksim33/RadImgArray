import nibabel as nib
import numpy as np
from radimgarray import SegImageArray

from .test_rad_img_array import behaves_like_ndarray


def test_seg_image_array(nifti_seg_file):
    seg = SegImageArray(nifti_seg_file)
    nii = nib.load(nifti_seg_file)
    np_array = nii.get_fdata()
    behaves_like_ndarray(seg, np_array)


def test_get_seg_indices(nifti_seg_file):
    seg = SegImageArray(nifti_seg_file)
    nii = nib.load(nifti_seg_file)
    np_array = nii.get_fdata()
    for value in seg.seg_values:
        indices = list(zip(*np.where(np_array == value)))
        seg_indices = seg.get_seg_indices(value)
        assert seg_indices == indices
