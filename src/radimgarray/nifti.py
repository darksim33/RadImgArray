from __future__ import annotations
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path


def load(file: Path):
    """
    Load a nifti file
    Args:
        file:

    Returns:
        :nifti1.Nifti1Image | :nifti2.Nifti2Image
    """
    return nib.load(file)


def update(
    array: np.ndarray | list, nii: nib.Nifti1Image | nib.Nifti2Image, **kwargs
) -> nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image:
    """
    Update nifti data
    Args:
        array:
        nii:
        **kwargs:

    Returns:

    """
    if nii is not None:
        return nib.Nifti1Image(array, nii.affine, nii.header)
    else:
        affine = kwargs.get("affine", np.eye(4))
        header = kwargs.get("header", nib.Nifti1Header())
        if isinstance(kwargs.get("dtype"), float):
            header.set_data_dtype("i4")
        elif isinstance(kwargs.get("dtype"), int):
            header.set_data_dtype("f4")

        if kwargs.get("nifti_dtype", "Nifti1") == "Nifti2":
            if isinstance(header, nib.nifti1.Nifti1Header):
                header = nib.nifti2.Nifti2Header.from_header(header)
            return nib.Nifti2Image(array, affine, header)
        else:
            return nib.Nifti1Image(array, affine, header)


def save(
    array: np.ndarray,
    path: Path,
    nii: nib.Nifti1Image | None = None,
    do_zip: bool = True,
):
    """
    Save array to nifti file
    Args:
        array: numpy array
        nii: nibabel image type
        path: path to save nifti file
        do_zip: whether to zip the nifti file
    """
    nii_array = nii.get_fdata()
    if nii_array.shape != array.shape:
        warnings.warn(
            "Array dimensions have changed since import. Updating nifti header."
        )
        nii = update(array, nii)

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
