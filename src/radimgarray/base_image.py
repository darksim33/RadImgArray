from __future__ import annotations
import numpy as np
from pathlib import Path
from copy import deepcopy
import nibabel as nib

from . import nifti
from . import plotting
from . import dicom


class RadImgArray(np.ndarray):
    info: dict
    def __new__(cls, _input: np.ndarray | list | Path | str, *args, **kwargs):
        if isinstance(_input, (Path, str)):
            _input = Path(_input) if isinstance(_input, str) else _input
            array, info = cls.__load(_input, args, kwargs)
        elif isinstance(_input, list):
            array = np.array(_input)
            info = {"type": "list"}
        elif isinstance(_input, np.ndarray):
            array = _input
            info = {"type": "np_array"}
        elif _input is None:
            raise TypeError("RadImgArray() missing required argument 'input' (pos 0)")
        else:
            raise TypeError("Input type not supported")
        obj = np.asarray(array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, "info", {"type": None})


    @classmethod
    def __load(cls, path: Path, *args, **kwargs) -> np.array:
        """
        Load image data from file. Either Dicom or NifTi are supported.
        Args:
            path: Path to image file or folder
            *args:
            **kwargs:
        """
        if nifti.check_for_nifti(path):
            return nifti.load(path)
        elif path.suffix == ".dcm" or path.is_dir():
            return dicom.load(path)

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
            nifti.save(np_array, path, self.info, **kwargs)
        elif path.suffix == ".dcm" or path.is_dir():
            dicom.save(self, path, self.info)

    def show(self):
        plotting.show_image(self)
