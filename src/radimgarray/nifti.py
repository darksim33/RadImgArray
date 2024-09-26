import numpy as np
import nibabel as nib
from pathlib import Path


def load(path: Path):
    return nib.load(path)


def update(data: np.ndarray | list, nii: nib.Nifti1Image | nib.Nifti2Image, **kwargs):
    if nii is not None:
        return nib.Nifti1Image(data, nii.affine, nii.header)
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
            return nib.Nifti2Image(data, affine, header)
        else:
            return nib.Nifti1Image(data, affine, header)


def save(nii: nib.Nifti1Image, path: Path, do_zip: bool = True):
    if do_zip and path.suffix != ".gz":
        nib.save(nii, path.with_suffix(".nii.gz"))
    else:
        nib.save(nii, path)


def check_for_nifti(file: Path):
    if file.is_file():
        if file.suffix == ".nii":
            return True
        elif file.suffix == ".gz":
            if Path(file.stem).suffix == ".nii":
                return True

def dicom_header_to_nifti_header(dicom_headers):
    nifti_header = nib.Nifti1Header()
    for dcm in dicom_headers:
        for key, value in dcm.items():
            if key not in nifti_header:
                try:
                    nifti_header[key] = value
                except KeyError:
                    pass
            elif header[key] != value:
                if type(header[key]) != list:
                    nifti_header[key] = list(header[key])
                header[key].append(value)
    return nifti_header