import nibabel as nib
import numpy as np
import pytest

from radimgarray import ImgArray

from .test_rad_img_array import behaves_like_ndarray


class TestNiftiLoading:
    """Test suite for NIfTI file loading functionality."""

    def test_load_from_nifti_file_matches_nibabel(self, nifti_file):
        """Test loading from NIfTI file produces array matching nibabel output."""
        rad_img = ImgArray(nifti_file)
        
        nii = nib.load(nifti_file)
        np_array = nii.get_fdata()
        behaves_like_ndarray(rad_img, np_array)

    def test_load_from_nifti_raw_data_matches_nibabel(self, nifti_file):
        """Test raw data from NIfTI file exactly matches nibabel fdata."""
        img = ImgArray(nifti_file)
        nii = nib.load(nifti_file)
        array = nii.get_fdata()
        assert np.allclose(img, array)

    def test_load_from_nonexistent_file_raises_error(self, root):
        """Test loading from non-existent NIfTI file raises FileNotFoundError."""
        nonexistent_file = root / "nonexistent.nii.gz"
        with pytest.raises(FileNotFoundError):
            ImgArray(nonexistent_file)

    def test_load_preserves_nifti_metadata(self, nifti_file):
        """Test loading NIfTI file preserves important metadata attributes."""
        rad_img = ImgArray(nifti_file)
        nii = nib.load(nifti_file)
        
        # Check that basic attributes are preserved in info dict
        assert hasattr(rad_img, 'info')
        assert 'affine' in rad_img.info
        assert 'header' in rad_img.info


class TestNiftiSaving:
    """Test suite for NIfTI file saving functionality."""

    def test_save_creates_file_at_specified_path(self, nifti_file, nifti_out_file):
        """Test saving ImgArray creates NIfTI file at specified output path."""
        rad_img = ImgArray(nifti_file)
        rad_img.save(nifti_out_file)
        assert nifti_out_file.exists()

    def test_save_and_reload_preserves_data(self, nifti_file, nifti_out_file):
        """Test saved and reloaded NIfTI file contains identical data."""
        original = ImgArray(nifti_file)
        original.save(nifti_out_file)
        
        reloaded = ImgArray(nifti_out_file)
        assert np.allclose(original, reloaded)

    def test_save_overwrites_existing_file(self, nifti_file, nifti_out_file):
        """Test saving to existing file path overwrites the file successfully."""
        rad_img = ImgArray(nifti_file)
        
        # Save once
        rad_img.save(nifti_out_file)
        first_data = ImgArray(nifti_out_file)
        
        # Modify and save again
        modified = rad_img * 2
        modified.save(nifti_out_file)
        second_data = ImgArray(nifti_out_file)
        
        # Verify data was overwritten
        assert not np.allclose(first_data, second_data)
        assert np.allclose(modified, second_data)

    def test_save_to_invalid_directory_raises_error(self, nifti_file, root):
        """Test saving to non-existent directory raises appropriate error."""
        rad_img = ImgArray(nifti_file)
        invalid_path = root / "nonexistent_dir" / "out.nii.gz"
        
        with pytest.raises((FileNotFoundError, OSError)):
            rad_img.save(invalid_path)

    def test_save_with_nii_gz_extension_succeeds(self, nifti_file, root):
        """Test saving with .nii.gz extension succeeds."""
        rad_img = ImgArray(nifti_file)
        out_file = root / "tests" / "out_img.nii.gz"
        
        try:
            rad_img.save(out_file)
            assert out_file.exists()
        finally:
            if out_file.exists():
                out_file.unlink()
