import pytest
from pathlib import Path

@pytest.fixture
def root():
    return Path(__file__).parent.parent

@pytest.fixture
def nifti_file():
    return Path(__file__).parent / ".assets/test_img.nii.gz"