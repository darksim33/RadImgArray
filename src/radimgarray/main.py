import numpy as np
from pathlib import Path
from copy import deepcopy

from . import nifti
from . import plotting
from . import dicom


class RadImgArray(np.ndarray):
    def __new__(cls, input_array: np.ndarray | list | None = None, *args, **kwargs):
        if input_array is None:
            input_array = np.array([])
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array: np.ndarray | list | None = None, *args, **kwargs):
        super().__init__()
        self.path: Path | None = None
        self.nifti: nibabel.nifti1.Nifti1Image | nibabel.nifti1.Nifti1Image | None = (
            None
        )
        self.dicom = dicom.DicomImage()

    def __array_finalize__(self, obj, /):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    def load(self, path: Path | str, **kwargs):

        self.path = path if isinstance(path, Path) else Path(path)

        if self.path.suffix == ".nii":
            if self.path.is_file():
                self.nifti = nifit.load(path)
                self.update_data(self.nifti.get_fdata())
        elif self.path.suffix == ".dcm" or self.path.is_dir():
            # TODO add dcm importer
            self.dicom.data, self.dicom.header = dicom.load(self.path)
            self.update_data(self.dicom.data)

    def update_data(self, data: np.ndarray | list):
        self[:] = data
        self.nifti = nifti.update(data, self.nifti)

    def copy(self, **kwargs):
        return deepcopy(self)

    def save(self, path: Path | str, **kwargs):
        path = path if isinstance(path, Path) else Path(path)
        if nifti.check_for_nifti(path):
            nifti.save(self.nifti, path)

    def show(self):
        plotting.show_image(self)
