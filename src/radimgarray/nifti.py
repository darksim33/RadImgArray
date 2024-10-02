from __future__ import annotations
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path

class NiftiImage:
    def __init__(self, file: Path = None):
        self.header = None  # original Header from Nifti file
        self.affine = None  # original Affine from Nifti file (position of image array data in reference space)
        self.shape = None   # original shape from Nifti file

def load(file: Path) -> (np.ndarray, dict):
    """
    Load a nifti file
    Args:
        file:

    Returns:
        np.ndarray
    """
    img = nib.load(file)

    data = img.get_fdata()
    info = {
        "type": "nifti",
        "path": file,
        "header": img.header,
        "affine": img.affine,
        "shape": data.shape
    }
    return data, info

def save(
        array: np.ndarray | list,
        path: Path,
        info: dict,
        do_zip: bool = True,
        **kwargs
):
    """
    Save array to nifti file.
    Args:
        array: numpy array
        path: path to save nifti file
        info: dict with header and data information
        do_zip: whether to zip the nifti file
        kwargs:
            header: str header (default: None) Change header
            affine: numpy array (default: None) Change affine
            dtype: numpy dtype (default: None) Change dtype
    """

    if array.shape != info["shape"]:
        warnings.warn(
            "Array dimensions have changed since import. Updating nifti header."
        )

    if info["header"] is None:
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
    else: affine = info["affine"]

    if isinstance(kwargs.get("dtype"), float):
        header.set_data_dtype("f4")
    elif isinstance(kwargs.get("dtype"), int):
        header.set_data_dtype("i4")

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


def check_for_nifti(file: Path):
    if not file.is_dir():
        if file.suffix == ".nii":
            return True
        elif file.suffix == ".gz":
            if Path(file.stem).suffix == ".nii":
                return True
    else:
        return False


def dicom_header_to_nifti_header(dicom_headers):
    nifti_header = nib.Nifti1Header()
    for dcm in dicom_headers:
        for key, value in dcm.items():
            if key not in nifti_header:
                try:
                    nifti_header[key] = value
                except KeyError:
                    pass
            elif nifti_header[key] != value:
                if type(nifti_header[key]) != list:
                    nifti_header[key] = list(nifti_header[key])
                nifti_header[key].append(value)
    return nifti_header
