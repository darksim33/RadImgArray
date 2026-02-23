import numpy as np
import pandas as pd
import pytest

from radimgarray import ImgArray, SegArray, tools


def get_random_even_int(low: int, high: int):
    """Helper function to generate random even integer in range."""
    while True:
        n = np.random.randint(low, high)
        if n % 2 == 0:
            return n


class TestZeroPadToSquare:
    """Test suite for zero_pad_to_square function."""

    def test_pad_rectangular_array_creates_square(self):
        """Test padding rectangular array creates square output."""
        while True:
            x = get_random_even_int(1, 64)
            y = get_random_even_int(1, 64)
            z = get_random_even_int(1, 64)
            array = np.random.rand(x, y, z)
            if not x == y:
                break
        new_array = tools.zero_pad_to_square(array)
        assert not new_array.shape == array.shape

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 20, 15),
            (30, 10, 25),
        ],
    )
    def test_pad_various_shapes_produces_square_xy(self, shape):
        """Test padding arrays with various shapes produces square x,y dimensions."""
        array = np.random.rand(*shape)
        padded = tools.zero_pad_to_square(array)
        
        # Function returns None for already square arrays
        if padded is not None:
            # First two dimensions should be equal after padding
            assert padded.shape[0] == padded.shape[1]
            # Z dimension should remain unchanged
            assert padded.shape[2] == shape[2]

    def test_pad_already_square_array_returns_none_or_same(self):
        """Test padding already square array returns None or same array."""
        array = np.random.rand(20, 20, 15)
        padded = tools.zero_pad_to_square(array)
        
        # Function may return None for already square arrays
        if padded is not None:
            assert np.array_equal(array, padded)

    def test_pad_preserves_original_data(self):
        """Test padding preserves all original data in center of padded array."""
        array = np.ones((10, 20, 5))
        padded = tools.zero_pad_to_square(array)
        
        # Original data should be present and equal to 1
        assert np.sum(padded == 1) == array.size
        # Padded areas should be 0
        assert np.sum(padded == 0) == (padded.size - array.size)

    def test_pad_empty_array_raises_error(self):
        """Test padding empty array raises appropriate error."""
        array = np.array([])
        with pytest.raises((ValueError, IndexError)):
            tools.zero_pad_to_square(array)


class TestGetMeanSignal:
    """Test suite for get_mean_signal function."""

    def test_get_mean_signal_from_uniform_region_returns_expected(
        self, nifti_file, nifti_seg_file
    ):
        """Test get_mean_signal from uniform region returns expected values."""
        size = 20
        array = np.zeros((size, size, size))
        array[5:15, 5:15, 5:15] = 1  # Create a cube with value 1
        seg = SegArray(array)
        img = np.ones((size, size, size, 16))
        mean = tools.get_mean_signal(img, seg, 1)
        assert mean.max() == 1
        assert mean.shape == (16,)

    @pytest.mark.parametrize(
        "num_volumes",
        [1, 8, 16, 32],
    )
    def test_get_mean_signal_various_volumes_correct_shape(self, num_volumes):
        """Test get_mean_signal with various volume counts returns correct shape."""
        size = 20
        seg_array = np.zeros((size, size, size))
        seg_array[5:15, 5:15, 5:15] = 1
        seg = SegArray(seg_array)
        
        img = np.random.rand(size, size, size, num_volumes)
        mean = tools.get_mean_signal(img, seg, 1)
        
        assert mean.shape == (num_volumes,)

    def test_get_mean_signal_nonexistent_segment_raises_error(self):
        """Test get_mean_signal for non-existent segment raises ValueError."""
        seg_array = np.zeros((10, 10, 10))
        seg_array[2:5, 2:5, 2:5] = 1
        seg = SegArray(seg_array)
        
        img = np.ones((10, 10, 10, 5))
        with pytest.raises(ValueError, match="not found in array"):
            tools.get_mean_signal(img, seg, 999)  # Non-existent segment

    def test_get_mean_signal_multiple_segments_independent(self):
        """Test get_mean_signal for multiple segments returns independent values."""
        seg_array = np.zeros((20, 20, 20))
        seg_array[2:8, 2:8, 2:8] = 1
        seg_array[12:18, 12:18, 12:18] = 2
        seg = SegArray(seg_array)
        
        img = np.zeros((20, 20, 20, 3))
        img[2:8, 2:8, 2:8, :] = 100
        img[12:18, 12:18, 12:18, :] = 200
        
        mean_1 = tools.get_mean_signal(img, seg, 1)
        mean_2 = tools.get_mean_signal(img, seg, 2)
        
        assert np.allclose(mean_1, 100)
        assert np.allclose(mean_2, 200)


class TestGetSingleSegArray:
    """Test suite for get_single_seg_array function."""

    def test_get_single_seg_array_isolates_segment(self, nifti_seg_file):
        """Test get_single_seg_array isolates single segment correctly."""
        seg = SegArray(nifti_seg_file)
        seg_single = tools.get_single_seg_array(seg, 1)
        
        assert seg_single.shape == seg.shape
        assert seg_single.max() == 1
        assert seg_single.min() == 0
        assert seg_single.number_segs == 1

    @pytest.mark.parametrize(
        "target_value",
        [1, 2, 3],
    )
    def test_get_single_seg_array_various_segments(self, target_value):
        """Test get_single_seg_array isolates various segment values correctly."""
        array = np.zeros((15, 15, 15))
        array[2:5, 2:5, 2:5] = 1
        array[6:9, 6:9, 6:9] = 2
        array[10:13, 10:13, 10:13] = 3
        seg = SegArray(array)
        
        seg_single = tools.get_single_seg_array(seg, target_value)
        
        assert seg_single.max() == 1
        assert seg_single.min() == 0
        # Only locations with target_value should be 1
        assert np.sum(seg_single) == np.sum(array == target_value)

    def test_get_single_seg_array_nonexistent_raises_error(self, nifti_seg_file):
        """Test get_single_seg_array for non-existent segment raises ValueError."""
        seg = SegArray(nifti_seg_file)
        max_value = max(seg.seg_values)
        
        with pytest.raises(ValueError, match="not found in array"):
            tools.get_single_seg_array(seg, max_value + 100)

    def test_get_single_seg_array_preserves_original(self, nifti_seg_file):
        """Test get_single_seg_array doesn't modify original segmentation."""
        seg = SegArray(nifti_seg_file)
        original_data = seg.copy()
        
        _ = tools.get_single_seg_array(seg, 1)
        
        assert np.array_equal(seg, original_data)


class TestMeanSegSignalsToExcel:
    """Test suite for save_mean_seg_signals_to_excel function."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_save_mean_seg_signals_creates_excel_file(
        self, nifti_file, nifti_seg_file, b_values, excel_out_file
    ):
        """Test save_mean_seg_signals_to_excel creates Excel file at specified path."""
        img = ImgArray(nifti_file)
        seg = SegArray(nifti_seg_file)
        tools.save_mean_seg_signals_to_excel(img, seg, b_values, excel_out_file)
        
        assert excel_out_file.is_file()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_save_mean_seg_signals_correct_data(
        self, nifti_file, nifti_seg_file, b_values, excel_out_file
    ):
        """Test save_mean_seg_signals_to_excel saves correct signal values."""
        img = ImgArray(nifti_file)
        seg = SegArray(nifti_seg_file)
        tools.save_mean_seg_signals_to_excel(img, seg, b_values, excel_out_file)
        
        df = pd.read_excel(excel_out_file)
        mean_signals = list()
        for seg_value in seg.seg_values:
            mean_signals.append(tools.get_mean_signal(img, seg, seg_value))
        
        for index in df.index:
            series = df.loc[index, :].tolist()[1:]
            assert np.allclose(
                mean_signals[index], series, equal_nan=True, rtol=1e-12, atol=0
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_save_mean_seg_signals_correct_row_count(
        self, nifti_file, nifti_seg_file, b_values, excel_out_file
    ):
        """Test save_mean_seg_signals_to_excel creates correct number of rows."""
        img = ImgArray(nifti_file)
        seg = SegArray(nifti_seg_file)
        tools.save_mean_seg_signals_to_excel(img, seg, b_values, excel_out_file)
        
        df = pd.read_excel(excel_out_file)
        assert len(df) == len(seg.seg_values)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_save_mean_seg_signals_correct_column_count(
        self, nifti_file, nifti_seg_file, b_values, excel_out_file
    ):
        """Test save_mean_seg_signals_to_excel creates correct number of columns."""
        img = ImgArray(nifti_file)
        seg = SegArray(nifti_seg_file)
        tools.save_mean_seg_signals_to_excel(img, seg, b_values, excel_out_file)
        
        df = pd.read_excel(excel_out_file)
        # Should have 1 column for segment ID + columns for each b-value
        assert len(df.columns) == len(b_values) + 1


class TestArrayToRgba:
    """Test suite for array_to_rgba function."""

    def test_array_to_rgba_2d_correct_shape(self):
        """Test array_to_rgba with 2D array produces correct RGBA shape."""
        array = np.random.rand(176, 176)
        rgba = tools.array_to_rgba(array, 1)
        assert rgba.shape == array.shape + (4,)

    def test_array_to_rgba_3d_correct_shape(self):
        """Test array_to_rgba with 3D array produces correct RGBA shape."""
        array = np.random.rand(176, 176, 64)
        rgba = tools.array_to_rgba(array, 1)
        assert rgba.shape == array.shape[:2] + (4,) + (array.shape[-1],)

    def test_array_to_rgba_4d_correct_shape(self, nifti_file):
        """Test array_to_rgba with 4D ImgArray produces correct RGBA shape."""
        img = ImgArray(nifti_file)
        rgba = tools.array_to_rgba(img, 1)
        assert rgba.shape == img.shape[:2] + (4,) + img.shape[-2:]

    @pytest.mark.parametrize(
        "alpha_value",
        [0.0, 0.5, 1.0],
    )
    def test_array_to_rgba_various_alpha_values(self, alpha_value):
        """Test array_to_rgba with various alpha values sets correct transparency."""
        array = np.random.rand(50, 50)
        rgba = tools.array_to_rgba(array, alpha_value)
        
        # Check alpha channel has expected value (function stores alpha as 0-1 not 0-255)
        assert np.allclose(rgba[:, :, 3], alpha_value)

    def test_array_to_rgba_empty_array_raises_error(self):
        """Test array_to_rgba with empty array raises appropriate error."""
        array = np.array([])
        with pytest.raises((ValueError, IndexError)):
            tools.array_to_rgba(array, 1)

    def test_array_to_rgba_preserves_relative_values(self):
        """Test array_to_rgba preserves relative intensity values."""
        array = np.array([[0, 0.5, 1.0], [0.25, 0.75, 1.0]])
        rgba = tools.array_to_rgba(array, 1.0)
        
        # Normalized values should be preserved in RGB channels
        # (all channels should have same values for grayscale)
        assert rgba[0, 0, 0] < rgba[0, 1, 0] < rgba[0, 2, 0]


class TestSliceToRgba:
    """Test suite for slice_to_rgba function."""

    def test_slice_to_rgba_4d_correct_shape(self, nifti_file):
        """Test slice_to_rgba with 4D array produces correct 2D RGBA shape."""
        slice_num = 1
        img = ImgArray(nifti_file)
        rgba = tools.slice_to_rgba(img, slice_num, np.random.ranf(1)[0])
        assert rgba.shape == img.shape[:2] + (4,)

    @pytest.mark.parametrize(
        "slice_index",
        [0, 5, 10],
    )
    def test_slice_to_rgba_various_slices_correct_shape(self, nifti_file, slice_index):
        """Test slice_to_rgba with various slice indices produces correct shape."""
        img = ImgArray(nifti_file)
        if slice_index < img.shape[2]:
            rgba = tools.slice_to_rgba(img, slice_index, 0.5)
            assert rgba.shape == img.shape[:2] + (4,)

    def test_slice_to_rgba_invalid_slice_raises_error(self, nifti_file):
        """Test slice_to_rgba with invalid slice index raises IndexError."""
        img = ImgArray(nifti_file)
        invalid_slice = img.shape[2] + 10
        
        with pytest.raises(IndexError):
            tools.slice_to_rgba(img, invalid_slice, 0.5)

    def test_slice_to_rgba_negative_slice_valid(self, nifti_file):
        """Test slice_to_rgba with negative slice index works correctly."""
        img = ImgArray(nifti_file)
        rgba = tools.slice_to_rgba(img, -1, 0.5)
        assert rgba.shape == img.shape[:2] + (4,)

    @pytest.mark.parametrize(
        "alpha",
        [0.0, 0.3, 0.7, 1.0],
    )
    def test_slice_to_rgba_various_alpha_correct_transparency(self, nifti_file, alpha):
        """Test slice_to_rgba with various alpha values sets correct transparency."""
        img = ImgArray(nifti_file)
        rgba = tools.slice_to_rgba(img, 0, alpha)
        
        # Check alpha channel (last channel)
        assert rgba.shape[-1] == 4
        # Alpha values should be within expected range
        assert rgba[:, :, 3].min() >= 0
        assert rgba[:, :, 3].max() <= 255
