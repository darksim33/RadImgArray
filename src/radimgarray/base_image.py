from __future__ import annotations
import numpy as np
from pathlib import Path
from copy import deepcopy
import nibabel as nib

from . import nifti
from . import plotting
from . import dicom


class RadImgArray(np.ndarray):
    _nifti: _nifti.NiftiImage | None = (
        None  # stores initial nifti data if given
    )
    _dicom: _dicom.DicomImage | None = None  # stores initial dicom data if given
    _path: Path | None = None  # stores initial path if given

    def __new__(cls, _input: np.ndarray | list | Path | str, *args, **kwargs):
        # prepare subclasses
        cls._nifti = nifti.NiftiImage()
        cls._dicom = dicom.DicomImage()
        cls._path = None
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
        obj.nifti = cls._nifti
        obj.dicom = cls._dicom
        obj.path = cls._path
        return cls

    # def __init__(self, *args, **kwargs):
    #     # super().__init__()
    #     self.nifti = self._nifti
    #     self.path = self._path

    def __array_finalize__(self, obj):
        if obj is None: return
        self.dicom = getattr(obj, "dicom", None)
        self.nifti = getattr(obj, "nifti", None)
        self.path = getattr(obj, "path", None)


    @classmethod
    def _load(cls, path: Path, *args, **kwargs) -> np.array:
        """
        Load image data from file. Either Dicom or NifTi are supported.
        Args:
            path: Path to image file or folder
            *args:
            **kwargs:
        """
        cls._path = path
        if nifti.check_for_nifti(cls._path):
            return cls._nifti.load(path)
        elif cls._path.suffix == ".dcm" or cls._path.is_dir():
            return cls.dicom.load(cls._path)

    # @property
    # def dicom(self):
    #     return self._dicom
    #
    # @property
    # def path(self):
    #     return self._path

    def copy(self, **kwargs):
        """Copy array and metadata"""
        return deepcopy(self)

    def save(self, path: Path | str, save_as: str | None = None, **kwargs):
        path = path if isinstance(path, Path) else Path(path)
        if save_as in ["nifti", "nii", ".nii.gz", "NIfTI"] or (
                _nifti.check_for_nifti(path) and not save_as in ["dicom", "dcm", "DICOM"]
        ):
            # TODO: Update nifti data if necessary
            np_array = np.array(self.copy())
            self._nifti.save(np_array, path, **kwargs)
        elif path.suffix == ".dcm" or path.is_dir():
            _dicom.save(self._dicom, path)

    def show(self):
        plotting.show_image(self)
