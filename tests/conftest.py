import shutil
from pathlib import Path

import pytest


@pytest.fixture
def root():
    return Path(__file__).parent.parent


@pytest.fixture
def tmpdir(root):
    return root / "tests" / ".temp"


@pytest.fixture
def nifti_file(root):
    return root / "tests" / ".assets" / "test_img.nii.gz"


@pytest.fixture
def nifti_out_file(tmpdir):
    file = tmpdir / "out_img.nii.gz"
    yield file
    if file.is_file():
        file.unlink()


@pytest.fixture
def nifti_seg_file(root):
    return root / "tests" / ".assets" / "test_seg.nii.gz"


@pytest.fixture
def dicom_folder(root):
    return root / "tests" / ".assets" / "dicom"


@pytest.fixture
def dicom_out_folder(tmpdir):
    """Fixture for DICOM output folder with cleanup."""
    folder = tmpdir / "out_dicom"
    yield folder
    if folder.exists():
        shutil.rmtree(folder)


@pytest.fixture
def excel_out_file(tmpdir):
    file = tmpdir / "out.xlsx"
    yield file
    if file.is_file():
        file.unlink()


@pytest.fixture
def b_values(root):
    file = root / "tests" / ".assets" / "test_bvalues.bval"
    with file.open("r") as f:
        return [int(value) for value in f.read().split("\n")]
