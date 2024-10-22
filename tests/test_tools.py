import numpy as np
from radimgarray import tools
from radimgarray import SegImageArray


def get_random_even_int():
    while True:
        n = np.random.randint(1, 64)
        if n % 2 == 0:
            return n


def test_zero_pad_to_square():
    while True:
        x = get_random_even_int()
        y = get_random_even_int()
        z = get_random_even_int()
        array = np.random.rand(x, y, z)
        if not x == y:
            break
    new_array = tools.zero_pad_to_square(array)
    assert not new_array.shape == array.shape


def test_get_mean_signal(nifti_file, nifti_seg_file):
    cube_size = np.random.randint(20, 64)
    size = np.random.randint(10, 32) * 2 + cube_size * 2
    array = np.zeros((size, size, size, 1))
    array[
        cube_size:-cube_size,
        cube_size:-cube_size,
        cube_size:-cube_size,
    ] = 1
    seg = SegImageArray(array)
    img = np.ones((size, size, size, 16))
    mean = tools.get_mean_signal(img, seg, 1)
    assert mean.max == 1
