import pytest
from pathlib import Path


@pytest.fixture
def root():
    return Path(__file__).parent.parent


@pytest.fixture
def nifti_file(root):
    return root / "tests" / ".assets" / "test_img.nii.gz"


@pytest.fixture
def nifti_out_file(root):
    file = root / "tests" / "out_img.nii.gz"
    yield file
    if file.is_file():
        file.unlink()


@pytest.fixture
def dicom_folder(root):
    return root / "tests" / ".assets" / "dicom"
