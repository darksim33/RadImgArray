import numpy as np
import pytest

from radimgarray import ImgArray


def behaves_like_ndarray(rad_img: ImgArray, np_array: np.ndarray):
    """Helper function to verify ImgArray behaves like numpy ndarray."""
    # Check if shapes are the same
    assert rad_img.shape == np_array.shape

    # Check if all elements are the same
    assert np.allclose(rad_img, np_array)

    # Check if max values are the same
    assert rad_img.max() == np_array.max()

    # Check if sum values are the same
    assert rad_img.sum() == np_array.sum()

    # Check if mean values are the same
    assert rad_img.mean() == np_array.mean()

    # Check if slicing works the same way
    assert np.allclose(rad_img[1], np_array[1])
    assert np.allclose(rad_img[:, 1], np_array[:, 1])
    assert np.allclose(rad_img[1, :, 1], np_array[1, :, 1])

    # Check if reshaping works the same way
    assert np.allclose(
        rad_img.reshape(
            rad_img.shape[0] * rad_img.shape[-1], rad_img.shape[1], rad_img.shape[2]
        ),
        np_array.reshape(
            np_array.shape[0] * np_array.shape[-1], np_array.shape[1], np_array.shape[2]
        ),
    )

    # Check if transposing works the same way
    assert np.allclose(rad_img.transpose(), np_array.transpose())


class TestImgArray:
    """Test suite for ImgArray class."""

    def test_create_from_array_valid_4d_success(self):
        """Test creating ImgArray from valid 4D numpy array succeeds."""
        np_array = np.random.rand(3, 3, 3, 3)
        rad_img = ImgArray(np_array)
        behaves_like_ndarray(rad_img, np_array)

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 10, 10, 5),
            (64, 64, 32, 16),
            (128, 128, 64, 3),
        ],
    )
    def test_create_from_array_various_shapes_success(self, shape):
        """Test creating ImgArray from numpy arrays with various valid shapes."""
        np_array = np.random.rand(*shape)
        rad_img = ImgArray(np_array)
        behaves_like_ndarray(rad_img, np_array)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            None,
            "not an array",
            123,
            {"not": "array"},
        ],
    )
    def test_create_from_invalid_input_raises_error(self, invalid_input):
        """Test creating ImgArray from invalid input types raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            ImgArray(invalid_input)

    def test_create_from_array_preserves_dtype(self):
        """Test creating ImgArray preserves the numpy array data type."""
        for dtype in [np.float32, np.float64, np.int32]:
            np_array = np.random.rand(5, 5, 5, 2).astype(dtype)
            rad_img = ImgArray(np_array)
            assert rad_img.dtype == dtype

    def test_slicing_operations_match_numpy(self):
        """Test various slicing operations match numpy array behavior."""
        np_array = np.random.rand(10, 10, 10, 5)
        rad_img = ImgArray(np_array)
        
        # Test single index
        assert np.allclose(rad_img[0], np_array[0])
        
        # Test range slicing
        assert np.allclose(rad_img[1:5], np_array[1:5])
        
        # Test multi-dimensional slicing
        assert np.allclose(rad_img[:, 5, :, 2], np_array[:, 5, :, 2])
        
        # Test negative indexing
        assert np.allclose(rad_img[-1], np_array[-1])

    def test_mathematical_operations_match_numpy(self):
        """Test mathematical operations produce same results as numpy array."""
        np_array = np.random.rand(5, 5, 5, 3)
        rad_img = ImgArray(np_array)
        
        # Test basic statistics
        assert np.allclose(rad_img.std(), np_array.std())
        assert np.allclose(rad_img.var(), np_array.var())
        assert np.allclose(rad_img.min(), np_array.min())
        
        # Test element-wise operations
        assert np.allclose(rad_img * 2, np_array * 2)
        assert np.allclose(rad_img + 1, np_array + 1)

    def test_reshape_operations_match_numpy(self):
        """Test reshape operations match numpy array behavior."""
        np_array = np.random.rand(4, 4, 4, 2)
        rad_img = ImgArray(np_array)
        
        new_shape = (8, 4, 2, 2)
        assert np.allclose(rad_img.reshape(new_shape), np_array.reshape(new_shape))

    def test_transpose_operations_match_numpy(self):
        """Test transpose operations match numpy array behavior."""
        np_array = np.random.rand(3, 4, 5, 2)
        rad_img = ImgArray(np_array)
        
        assert np.allclose(rad_img.T, np_array.T)
        assert np.allclose(rad_img.transpose(), np_array.transpose())
        assert np.allclose(rad_img.transpose(3, 2, 1, 0), np_array.transpose(3, 2, 1, 0))
