import numpy as np
from radimgarray import tools
from radimgarray import SegImageArray


def get_random_even_int(low: int, high: int):
    while True:
        n = np.random.randint(low, high)
        if n % 2 == 0:
            return n


def test_zero_pad_to_square():
    while True:
        x = get_random_even_int(1, 64)
        y = get_random_even_int(1, 64)
        z = get_random_even_int(1, 64)
        array = np.random.rand(x, y, z)
        if not x == y:
            break
    new_array = tools.zero_pad_to_square(array)
    assert not new_array.shape == array.shape


def test_get_mean_signal(nifti_file, nifti_seg_file):
    cube_size = get_random_even_int(10, 32)
    size = get_random_even_int(10, 64) + cube_size
    array = np.zeros((size, size, size, 1))
    array[
        cube_size:-cube_size,
        cube_size:-cube_size,
        cube_size:-cube_size,
    ] = 1
    seg = SegImageArray(array)
    img = np.ones((size, size, size, 16))
    mean = tools.get_mean_signal(img, seg, 1)
    assert mean.max() == 1
