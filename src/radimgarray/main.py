import numpy as np
from pathlib import Path
from copy import deepcopy
import nibabel as nib

from . import nifti
from . import plotting
from . import dicom


class RadImgArray(np.ndarray):
    _nifti: nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image | None = (
        None  # stores initial nifti data if given
    )
    _dicom: dicom.DicomImage | None = None  # stores initial dicom data if given
    _path: Path | None = None  # stores initial path if given

    def __new__(cls, _input: np.ndarray | list | Path | str, *args, **kwargs):
        if isinstance(_input, (Path, str)):
            _input = Path(_input) if isinstance(_input, str) else _input
            _input = cls._load(_input, args, kwargs)
        elif isinstance(_input, list):
            _input = np.array(_input)
        elif isinstance(_input, np.ndarray):
            pass
        elif _input is None:
            raise TypeError("RadImgArray() missing required argument 'input' (pos 0)")
        else:
            raise TypeError("Input type not supported")
        obj = np.asarray(_input).view(cls)
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __array_finalize__(self, obj, /):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @classmethod
    def _load(cls, path: Path, *args, **kwargs) -> np.array:
        """
        Load image data from file. Either Dicom or NifTi are supported.
        Args:
            path: Path to image file or folder
            *args:
            **kwargs:
        """
        cls.path = path
        if nifti.check_for_nifti(cls.path):
            cls.nifti = nifti.load(path)
            return cls.nifti.get_fdata()
        elif cls.path.suffix == ".dcm" or cls.path.is_dir():
            cls.dicom = dicom.DicomImage()
            cls.dicom.data, cls.dicom.header = dicom.load(cls.path)
            return cls.dicom.data.copy()

    def copy(self, **kwargs):
        """Copy array and metadata"""
        return deepcopy(self)

    def save(self, path: Path | str, save_as: str | None = None, **kwargs):
        path = path if isinstance(path, Path) else Path(path)
        if save_as in ["nifti", "nii", ".nii.gz", "NIfTI"] or (
            nifti.check_for_nifti(path) and not save_as in ["dicom", "dcm", "DICOM"]
        ):
            # TODO: Update nifti data if necessary
            np_array = np.array(self.copy())
            nifti.save(np_array, path, self.nifti)
        elif path.suffix == ".dcm" or path.is_dir():
            dicom.save(self.dicom, path)

    def show(self):
        plotting.show_image(self)
