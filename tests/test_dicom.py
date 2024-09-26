from pathlib import Path

from radimgarray import RadImgArray


def test_load_dicom():
    folder = Path(r"E:\data\testdata\240926_NNLSPhantom")
    rad_img = RadImgArray().load(folder)
