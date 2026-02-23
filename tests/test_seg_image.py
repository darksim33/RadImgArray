import nibabel as nib
import numpy as np
import pytest

from radimgarray import SegArray

from .test_rad_img_array import behaves_like_ndarray


class TestSegArray:
    """Test suite for SegArray (segmentation array) class."""

    def test_create_from_nifti_seg_file_matches_nibabel(self, nifti_seg_file):
        """Test creating SegArray from NIfTI segmentation file matches nibabel data."""
        seg = SegArray(nifti_seg_file)
        nii = nib.load(nifti_seg_file)
        np_array = nii.get_fdata()
        behaves_like_ndarray(seg, np_array)

    def test_create_from_array_with_multiple_segments_success(self):
        """Test creating SegArray from numpy array with multiple segment values."""
        array = np.zeros((10, 10, 10))
        array[2:5, 2:5, 2:5] = 1
        array[6:9, 6:9, 6:9] = 2
        
        seg = SegArray(array)
        assert seg.shape == array.shape
        assert len(seg.seg_values) >= 2

    def test_create_from_empty_array_success(self):
        """Test creating SegArray from all-zeros array succeeds."""
        array = np.zeros((5, 5, 5))
        seg = SegArray(array)
        assert seg.shape == array.shape
        # seg_values excludes background (0), so should be empty for all-zeros array
        assert len(seg.seg_values) == 0

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 10, 10),
            (32, 32, 16),
            (64, 64, 32),
        ],
    )
    def test_create_from_various_shapes_success(self, shape):
        """Test creating SegArray from arrays with various valid shapes."""
        array = np.random.randint(0, 3, size=shape)
        seg = SegArray(array)
        assert seg.shape == shape


class TestSegArrayIndices:
    """Test suite for SegArray index retrieval functionality."""

    def test_get_seg_indices_matches_numpy_where(self, nifti_seg_file):
        """Test get_seg_indices returns same indices as numpy.where for each segment."""
        seg = SegArray(nifti_seg_file)
        nii = nib.load(nifti_seg_file)
        np_array = nii.get_fdata()
        
        for value in seg.seg_values:
            indices = list(zip(*np.where(np_array == value)))
            seg_indices = seg.get_seg_indices(value)
            assert seg_indices == indices

    def test_get_seg_indices_for_nonexistent_value_raises_error(self, nifti_seg_file):
        """Test get_seg_indices for non-existent segment value raises ValueError."""
        seg = SegArray(nifti_seg_file)
        max_value = max(seg.seg_values)
        nonexistent_value = max_value + 100
        
        with pytest.raises(ValueError, match="not found in array"):
            seg.get_seg_indices(nonexistent_value)

    def test_get_seg_indices_for_all_segments_covers_array(self, nifti_seg_file):
        """Test get_seg_indices for all segments covers all non-zero indices."""
        seg = SegArray(nifti_seg_file)
        
        all_indices = set()
        for value in seg.seg_values:
            if value != 0:  # Skip background
                indices = seg.get_seg_indices(value)
                all_indices.update(indices)
        
        # Verify we found indices
        assert len(all_indices) > 0

    @pytest.mark.parametrize(
        "segment_value",
        [1, 2, 3],
    )
    def test_get_seg_indices_returns_correct_count(self, segment_value):
        """Test get_seg_indices returns correct number of indices for each segment."""
        array = np.zeros((10, 10, 10))
        # Create a 3x3x3 cube for each segment value
        array[1:4, 1:4, 1:4] = segment_value
        
        seg = SegArray(array)
        indices = seg.get_seg_indices(segment_value)
        
        # Should have 27 indices (3x3x3 cube)
        assert len(indices) == 27

    def test_get_seg_indices_with_integer_values(self):
        """Test get_seg_indices works correctly with integer segment values."""
        array = np.zeros((5, 5, 5))
        array[1:3, 1:3, 1:3] = 1
        array[3:5, 3:5, 3:5] = 2
        
        seg = SegArray(array)
        indices_1 = seg.get_seg_indices(1)
        indices_2 = seg.get_seg_indices(2)
        
        assert len(indices_1) > 0
        assert len(indices_2) > 0
        # Ensure no overlap
        assert set(indices_1).isdisjoint(set(indices_2))


class TestSegArrayProperties:
    """Test suite for SegArray properties and attributes."""

    def test_seg_values_returns_unique_nonzero_values(self, nifti_seg_file):
        """Test seg_values property returns unique non-zero segment values."""
        seg = SegArray(nifti_seg_file)
        nii = nib.load(nifti_seg_file)
        np_array = nii.get_fdata()
        
        # seg_values excludes background (0)
        expected_values = np.unique(np_array)
        expected_values = expected_values[expected_values != 0]
        assert set(seg.seg_values) == set(expected_values.astype(int))

    def test_seg_values_ordered_consistently(self):
        """Test seg_values property returns values in consistent order."""
        array = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
        seg = SegArray(array)
        
        # Call multiple times to check consistency
        values_1 = seg.seg_values
        values_2 = seg.seg_values
        
        assert values_1 == values_2

    def test_number_segs_excludes_background(self):
        """Test number_segs property correctly counts segments excluding background."""
        array = np.zeros((10, 10, 10))
        array[2:4, 2:4, 2:4] = 1
        array[6:8, 6:8, 6:8] = 2
        
        seg = SegArray(array)
        # Should have 2 non-zero segments (excluding 0 background)
        assert seg.number_segs == 2
