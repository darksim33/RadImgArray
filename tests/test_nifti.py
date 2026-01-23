import nibabel as nib

from radimgarray import ImgArray

from .test_rad_img_array import behaves_like_ndarray


def test_from_nii(nifti_file):
    rad_img = ImgArray(nifti_file)

    nii = nib.load(nifti_file)
    np_array = nii.get_fdata()
    behaves_like_ndarray(rad_img, np_array)


def test_save_from_nii(nifti_file, nifti_out_file):
    rad_img = ImgArray(nifti_file)
    rad_img.save(nifti_out_file)
    assert nifti_out_file.exists()
