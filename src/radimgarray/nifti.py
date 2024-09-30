from __future__ import annotations
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path

class NiftiImage:
    def __init__(self, file: Path = None):
        self.header = None  # original Header from Nifti file
        self.affine = None  # original Affine from Nifti file
        self.shape = None   # original shape from Nifti file

    def load(self, file: Path) -> np.ndarray:
        """
        Load a nifti file
        Args:
            file:

        Returns:
            np.ndarray
        """
        img = nib.load(file)
        self.header = img.header
        self.affine = img.affine
        data = img.get_fdata()
        self.shape = data.shape
        return data

    def save(
            self,
            array: np.ndarray | list,
            path: Path,
            do_zip: bool = True,
            **kwargs
    ):
        """
        Save array to nifti file.
        Args:
            array: numpy array
            path: path to save nifti file
            do_zip: whether to zip the nifti file
            kwargs:
                header: str header (default: None) Change header
                affine: numpy array (default: None) Change affine
                dtype: numpy dtype (default: None) Change dtype
        """

        if array.shape != self.shape:
            warnings.warn(
                "Array dimensions have changed since import. Updating nifti header."
            )

        if self.header is None:
            if kwargs.get("header", "Nifti1") == "Nifti1":
                self.header = nib.nifti1.Nifti1Header()
            else:
                self.header = nib.nifti2.Nifti2Header()
        else:
            if kwargs.get("header", "Nifti1") == "Nifti2":
                if isinstance(self.header, nib.nifti1.Nifti1Header):
                    self.header = nib.nifti2.Nifti2Header.from_header(self.header)
            else:
                if not isinstance(self.header, nib.nifti1.Nifti1Header):
                    self.header = nib.nifti1.Nifti1Header.from_header(self.header)

        if self.affine is None:
            self.affine = kwargs.get("affine", np.eye(4))

        if isinstance(kwargs.get("dtype"), float):
            self.header.set_data_dtype("f4")
        elif isinstance(kwargs.get("dtype"), int):
            self.header.set_data_dtype("i4")

        if isinstance(self.header, nib.nifti1.Nifti1Header):
            nii = nib.Nifti1Image(array, self.affine, self.header)
        elif isinstance(self.header, nib.nifti2.Nifti2Header):
            nii = nib.Nifti2Image(array, self.affine, self.header)

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
