import numpy as np
from radimgarray import tools
from radimgarray import RadImgArray


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

    new_array_fft = tools.zero_pad_to_square(array, k_space=True)
    assert not new_array_fft.shape == array.shape
