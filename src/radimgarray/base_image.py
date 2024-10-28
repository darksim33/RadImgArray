"""Base image class for radiological enhanced image array

This module provides the base image class for radiological enhanced image array.

Classes:
    RadImgArray: Radiological enhanced image array class
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from copy import deepcopy

from . import nifti
from . import plotting
from . import dicom


class RadImgArray(np.ndarray):
    """Radiological enhanced image array class

    Attributes:
        info (dict): dictionary with additional information about the image like nifti
            or dicom headers.
            {
                "type": "nifti" | "dicom" | "list" | "np_array",
                "path": Path,
                "header": [{},...]
                "affine": np.eye(4),
                "shape": (x, y, z, t)
            }
    Methods:
        __new__: Create a new RadImgArray object
        __array_finalize__: Copy metadata when creating new array
        copy: Copy array and metadata
        save: Save image data to file
        show: Display image
    """

    info: dict

    def __new__(
        cls,
        _input: np.ndarray | list | Path | str,
        info: dict | None = None,
        *args,
        **kwargs,
    ):
        """Create a new RadImgArray object
        Args:
            _input (np.ndarray, list, Path, str): data input to transform/load
            info (dict, optional): dictionary from a already initialized RadImageArray
                with additional information about the image
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        if isinstance(_input, (Path, str)):
            _input = Path(_input) if isinstance(_input, str) else _input
            array, info = cls.__load(_input, args, kwargs)
        elif isinstance(_input, list):
            array = _input
            if info is not None:
                info = info
            else:
                info = cls.__get_default_info()
                info["type"] = "list"
        elif isinstance(_input, np.ndarray):
            array = _input
            if info is not None:
                info = info
            else:
                info = cls.__get_default_info()
                info["type"] = "np_array"
        elif _input is None:
            raise TypeError("RadImgArray() missing required argument 'input' (pos 0)")
        else:
            raise TypeError("Input type not supported")
        obj = np.asarray(array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", {"type": None})

    @classmethod
    def __load(cls, path: Path, *args, **kwargs) -> np.array:
        """
        Load image data from file. Either Dicom or NifTi are supported.
        Args:
            path (Path): to image file or folder
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
        """Save image data to file.

        Can save either as NifTi, Dicom (or numpy+dict).
        Args:
            path (Path, str): to save the image
            save_as (str, optional): format to save the image
            **kwargs: additional keyword
        """
        path = path if isinstance(path, Path) else Path(path)
        if save_as in ["nifti", "nii", ".nii.gz", "NIfTI"] or (
            nifti.check_for_nifti(path) and save_as not in ["dicom", "dcm", "DICOM"]
        ):
            # TODO: Update nifti data if necessary
            np_array = np.array(self.copy())
            nifti.save(np_array, path, self.info, **kwargs)
        elif path.suffix == ".dcm" or path.is_dir():
            dicom.save(self, path, self.info)

    def show(self):
        plotting.show_image(self)

    @staticmethod
    def __get_default_info():
        return {
            "type": None,
            "path": Path(),
            "header": [],
            "affine": np.eye(4),
            "shape": [0, 0, 0, 0],
        }
