import numpy as np
from radimgarray import tools
from radimgarray import RadImgArray


def test_zero_pad_to_square():
    while True:
        x = np.random.randint(1, 64)
        y = np.random.randint(1, 64)
        z = np.random.randint(1, 64)
        array = np.ones((x, y, z))
        if not x == y:
            break
    new_array = tools.zero_pad_to_square(array)
    assert not new_array.shape == array.shape
    RadImgArray(new_array)
