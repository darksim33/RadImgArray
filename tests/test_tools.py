import numpy as np
import pandas as pd
from radimgarray import tools, RadImgArray
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
    assert not new_array.shape == array.shape if new_array is not None else False


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
    assert mean.shape == (16,)


def test_get_single_seg_array(nifti_seg_file):
    seg = SegImageArray(nifti_seg_file)
    seg_single = tools.get_single_seg_array(seg, 1)
    assert seg_single.shape == seg.shape
    assert seg_single.max() == 1
    assert seg_single.min() == 0
    assert seg_single.number_segs == 1


def test_mean_seg_signals_to_excel(
    nifti_file, nifti_seg_file, b_values, excel_out_file
):
    img = RadImgArray(nifti_file)
    seg = SegImageArray(nifti_seg_file)
    tools.save_mean_seg_signals_to_excel(img, seg, b_values, excel_out_file)
    assert excel_out_file.is_file()
    df = pd.read_excel(excel_out_file)
    mean_signals = list()
    for seg_value in seg.seg_values:
        mean_signals.append(tools.get_mean_signal(img, seg, seg_value))
    for index in df.index:
        series = df.loc[index, :].tolist()[1:]
        assert [round(value, 12) for value in mean_signals[index].tolist()] == [
            round(value, 12) for value in series
        ]


def test_array_to_rgba(nifti_file):
    # 2D case
    array = np.random.rand(176, 176)
    rgba = tools.array_to_rgba(array, 1)
    assert rgba.shape == array.shape + (4,)
    # 3D case
    array = np.random.rand(176, 176, 64)
    rgba = tools.array_to_rgba(array, 1)
    assert rgba.shape == array.shape[:2] + (4,) + (array.shape[-1],)
    # 4D case
    img = RadImgArray(nifti_file)
    rgba = tools.array_to_rgba(img, 1)
    assert rgba.shape == img.shape[:2] + (4,) + img.shape[-2:]


def test_slice_to_rgba(nifti_file):
    # 4D Case
    slice_num = 1
    img = RadImgArray(nifti_file)
    rgba = tools.slice_to_rgba(img, slice_num, np.random.ranf(1)[0])
    assert rgba.shape == img.shape[:2] + (4,)
