"""Nifti file handling functions.
Holds all functions to load, save and check nifti files.

Nifti info (dict):
{
    "type": "nifti",
    "path": Path,  # path to nifti file
    "header": img.header,  # header from nifti
    "affine": img.affine,  # affine from nifti or np.eye(4)
    "shape": (x, y, z, t),  # data.shape
}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np


def load(file: Path, **kwargs) -> tuple[np.ndarray, dict]:
    """Load a nifti file.

    Args:
        file (Path): path to nifti file
        kwargs:
            raw (bool): get data from dataobj.
    Returns:
        (np.ndarray, dict): nifti data and info
    """

    img = nib.load(file)
    if not kwargs.get("raw", False):
        data = img.get_fdata()
    else:
        data = np.array(img.dataobj)
    
    # apply dtype from header
    dtype = img.header.get_data_dtype()
    data = data.astype(dtype)
    
    info = {
        "type": "nifti",
        "path": file,
        "header": img.header,
        "affine": img.affine,
        "shape": data.shape,
    }
    return data, info


def save(
    array: np.ndarray | list, path: Path, info: dict, do_zip: bool = True, **kwargs
):
    """
    Save array to nifti file.
    Args:
        array (np.ndarray): data to save
        path (Path): to save nifti file
        info (dict): with header and data information
        do_zip (bool): whether to zip the nifti file
        kwargs:
            header (str): header (default: None) Change header
            affine (np.ndarray): array (default: None) Change affine
            dtype: numpy dtype (default: None) Change dtype
    """

    if array.shape != info["shape"] and kwargs.get("verbose", False):
        warnings.warn(
            "Array dimensions have changed since import. Updating nifti header."
        )

    if info["header"] is None or isinstance(info["header"], list):
        if kwargs.get("header", "Nifti1") == "Nifti1":
            header = nib.nifti1.Nifti1Header()
        else:
            header = nib.nifti2.Nifti2Header()
    else:
        header = info["header"]
        if kwargs.get("header", "Nifti1") == "Nifti2":
            if isinstance(header, nib.nifti1.Nifti1Header):
                header = nib.nifti2.Nifti2Header.from_header(header)
        else:
            if not isinstance(header, nib.nifti1.Nifti1Header):
                header = nib.nifti1.Nifti1Header.from_header(header)

    if info["affine"] is None:
        affine = kwargs.get("affine", np.eye(4))
    else:
        affine = info["affine"]

    dtype = kwargs.get("dtype", None)
    if dtype is float:
        header.set_data_dtype(np.float32)
    elif dtype is int:
        header.set_data_dtype(np.int32)

    if isinstance(header, nib.nifti1.Nifti1Header):
        nii = nib.Nifti1Image(array, affine, header)
    elif isinstance(header, nib.nifti2.Nifti2Header):
        nii = nib.Nifti2Image(array, affine, header)
    else:
        return
    if do_zip and path.suffix != ".gz":
        nib.save(nii, path.with_suffix(".nii.gz"))
    else:
        nib.save(nii, path)


def check_for_nifti(file: Path) -> bool:
    """Check if file is a nifti file.

    Args:
        file (Path): file to check
    Returns:
        bool: True if file is nifti file
    """
    if not file.is_dir():
        if file.suffix == ".nii":
            return True
        elif file.suffix == ".gz":
            if Path(file.stem).suffix == ".nii":
                return True
    else:
        return False


def dicom_header_to_nifti_header(dicom_headers: list):
    """Convert dicom headers to nifti headers.

    Args:
        dicom_headers (list): list of dicom headers (dict)
    """
    nifti_header = nib.Nifti1Header()
    for dcm in dicom_headers:
        for key, value in dcm.items():
            if key not in nifti_header:
                try:
                    nifti_header[key] = value
                except KeyError:
                    pass
            elif nifti_header[key] != value:
                if not isinstance(nifti_header[key], list):
                    nifti_header[key] = list(nifti_header[key])
                nifti_header[key].append(value)
    return nifti_header
