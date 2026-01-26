import numpy as np
import pytest

from radimgarray import ImgArray


class TestDicomLoading:
    """Test suite for DICOM file loading functionality."""

    def test_load_from_dicom_folder_creates_array(self, dicom_folder):
        """Test loading DICOM files from folder creates valid array."""
        rad_img = ImgArray(dicom_folder)
        
        # Should create a valid array
        assert rad_img is not None
        assert isinstance(rad_img, ImgArray)
        assert rad_img.size > 0

    def test_load_from_dicom_folder_correct_dimensions(self, dicom_folder):
        """Test loading DICOM files produces correct number of dimensions."""
        rad_img = ImgArray(dicom_folder)
        
        # Should be 3D or 4D
        assert rad_img.ndim in [3, 4]

    def test_load_from_dicom_folder_has_info(self, dicom_folder):
        """Test loading DICOM preserves metadata in info dict."""
        rad_img = ImgArray(dicom_folder)
        
        assert hasattr(rad_img, 'info')
        assert rad_img.info['type'] == 'dicom'
        assert 'header' in rad_img.info
        assert 'affine' in rad_img.info

    def test_load_from_nonexistent_folder_returns_none_or_raises(self, root):
        """Test loading from non-existent DICOM folder handles gracefully."""
        nonexistent_folder = root / "tests" / "nonexistent_dicom"
        
        try:
            result = ImgArray(nonexistent_folder)
            # If it doesn't raise, it should return None or empty
            assert result is None or result.size == 0
        except (FileNotFoundError, ValueError, TypeError):
            # These exceptions are acceptable
            pass

    def test_load_preserves_dicom_metadata(self, dicom_folder):
        """Test loading DICOM preserves important metadata attributes."""
        rad_img = ImgArray(dicom_folder)
        
        # Check that DICOM-specific attributes are present
        assert hasattr(rad_img, 'info')
        assert rad_img.info['type'] == 'dicom'
        assert isinstance(rad_img.info['header'], list)
        assert len(rad_img.info['header']) > 0

    def test_load_from_dicom_consistent_shape(self, dicom_folder):
        """Test loading same DICOM folder twice produces consistent shape."""
        rad_img_1 = ImgArray(dicom_folder)
        rad_img_2 = ImgArray(dicom_folder)
        
        assert rad_img_1.shape == rad_img_2.shape

    def test_load_dicom_data_type_valid(self, dicom_folder):
        """Test loaded DICOM data has valid numeric dtype."""
        rad_img = ImgArray(dicom_folder)
        
        assert rad_img.dtype in [np.float32, np.float64, np.int16, np.uint16, np.int32]


class TestDicomSaving:
    """Test suite for DICOM file saving functionality."""

    @pytest.mark.skip(reason="DICOM save functionality needs to be fully implemented")
    def test_save_creates_dicom_files(self, dicom_folder, dicom_out_folder):
        """Test saving ImgArray creates DICOM files in specified directory."""
        rad_img = ImgArray(dicom_folder)
        
        rad_img.save(dicom_out_folder)
        assert dicom_out_folder.exists()
        assert len(list(dicom_out_folder.glob("**/*.dcm"))) > 0

    @pytest.mark.skip(reason="DICOM save functionality needs to be fully implemented")
    def test_save_and_reload_preserves_shape(self, dicom_folder, dicom_out_folder):
        """Test saved and reloaded DICOM data has same shape."""
        original = ImgArray(dicom_folder)
        
        original.save(dicom_out_folder)
        reloaded = ImgArray(dicom_out_folder)
        
        assert reloaded.shape == original.shape

    @pytest.mark.skip(reason="DICOM save functionality needs to be fully implemented")
    def test_save_preserves_metadata(self, dicom_folder, dicom_out_folder):
        """Test saving DICOM preserves important metadata in saved files."""
        rad_img = ImgArray(dicom_folder)
        
        rad_img.save(dicom_out_folder)
        reloaded = ImgArray(dicom_out_folder)
        
        # Check key metadata is preserved
        assert reloaded.shape == rad_img.shape
        assert reloaded.info['type'] == 'dicom'


class TestDicomProperties:
    """Test suite for DICOM-specific properties and attributes."""

    def test_dicom_info_accessible(self, dicom_folder):
        """Test DICOM info dictionary is accessible after loading."""
        rad_img = ImgArray(dicom_folder)
        
        assert hasattr(rad_img, 'info')
        assert isinstance(rad_img.info, dict)
        
    def test_dicom_header_is_list(self, dicom_folder):
        """Test DICOM header is stored as list in info."""
        rad_img = ImgArray(dicom_folder)
        
        assert 'header' in rad_img.info
        assert isinstance(rad_img.info['header'], list)

    def test_dicom_affine_matrix_correct_shape(self, dicom_folder):
        """Test DICOM affine matrix has correct 4x4 shape."""
        rad_img = ImgArray(dicom_folder)
        
        assert 'affine' in rad_img.info
        assert rad_img.info['affine'].shape == (4, 4)

    def test_dicom_path_preserved(self, dicom_folder):
        """Test DICOM source path is preserved in info."""
        rad_img = ImgArray(dicom_folder)
        
        assert 'path' in rad_img.info
        assert rad_img.info['path'] == dicom_folder
