from __future__ import annotations
import numpy as np
import pydicom
from pathlib import Path


"""
Dicom Dict
{
    "type": "dicom",
    "path": Path,
    "header": [{},...]
    "affine": np.eye(4), not implemented jet
    "shape": (x, y, z, t)
}

"""

class DicomImage:
    def __init__(self):
        self.header = None  # list of dict with header information for each loaded dicom file
        # TODO: no method for actual calculation from header data available atm
        self.affine = np.eye(4)  # affine matrix for position of image array data in reference space
        self.shape = None

def load(path: Path) -> (np.ndarray | None, dict):
    """
    Load dicom files from a directory.
    If multiple series are discovered in the directory, the user is asked to select one.
    Args:
        path: folder containing dicom files or folders with files

    Returns:

    """
    dicom_data = []
    for file in path.glob("**/*"):
        if file.suffix == ".dcm" or file.suffix == ".dicom" or file.suffix == "":
            if file.is_file():
                try:
                    dicom_data.append(pydicom.dcmread(file))
                except pydicom.errors.InvalidDicomError:
                    # Not a dicom file
                    pass
    if dicom_data:
        dicom_series = get_series_data(dicom_data)
    else:
        return None

    dicom_series_sorted = sort_dicom_files(dicom_series)

    dicom_matrix = []
    for idx in range(len(dicom_series_sorted)):
        dicom_matrix.append([dcm.pixel_array for dcm in dicom_series_sorted[idx]])

    dicom_matrix = np.array(dicom_matrix)
    if dicom_matrix.ndim == 3:
        dicom_matrix = np.permute_dims(dicom_matrix, [1, 2, 0])
    if dicom_matrix.ndim == 4:
        dicom_matrix = np.permute_dims(dicom_matrix, [2, 3, 1, 0])
    info = {
        "type": "dicom",
        "path":path,
        "header": [dcm[0].items for dcm in dicom_series_sorted],
        "affine": np.eye(4),
        "shape": dicom_matrix.shape
    }
    return dicom_matrix, info

def save(array: np.ndarray | list, path: Path, info):
    """Save dicom data to path - simple copilot placeholder - untested"""
    if not (array.shape == info["shape"]).all():
        raise ValueError("Array dimensions have changed since import. Cannot save dicom files.")
    else:
        pass
        # permute data back to original shape and apply header
    for idx, dcm in enumerate(dicom.data):
        dcm.PixelData = dcm.pixel_array.tobytes()
        dcm.save_as(path / f"{idx}.dcm")
    return path




def get_series_data(series_list: list, interface: str = "cli") -> list:
    """
    Get series data from a list of dicom files
    Args:
        series_list: list of all found dicom files
        interface: select interface for user interaction when multiple dicom series are found

    Returns:
        single_series_list: list of dicom data from a single series
    """
    series_ids = []
    series_to_erase = []
    # some dicom file do not have SeriesInstanceUID
    for series in series_list:
        try:
            uid = series.SeriesInstanceUID
            series_ids.append(uid)
        except AttributeError:
            series_to_erase.append(series)

    # erase separately to avoid changing the list while iterating
    for series in series_to_erase:
        series_list.remove(series)

    def uniques(_list: list):
        unique_list = []
        for x in _list:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    unique_series_ids = uniques(series_ids)
    if len(unique_series_ids) > 1:
        if interface == "cli":
            print("Multiple series found in the directory.")
            for idx, series_id in enumerate(unique_series_ids):
                print(
                    f"[{idx}]: {series_list[series_ids.index(series_id)].SeriesDescription}"
                )
            number = input(
                f"Enter the number of the series you want to load [0-{len(unique_series_ids)-1}]: "
            )
            print(f"Loading series number {number}")
            series_id = unique_series_ids[int(number)]
        else:
            # Placeholder for GUI
            series_id = unique_series_ids[0]

    else:
        series_id = unique_series_ids[0]

    series_indexes = [idx for idx, x in enumerate(series_ids) if series_id == x]
    return series_list[series_indexes[0] : series_indexes[-1] + 1]


def sort_dicom_files(dicom_files: list[pydicom.Dataset]) -> list[list[pydicom.Dataset]]:
    """
    Sort dicom files by ImagePosition and 4. dimension functional data
    From IntervalLudger - GoNifti
    Args:
        dicom_files: unsorted list of dicom files
    Returns:
        : list of containing list of sorted dicom files
    """
    positions = {}
    for dicom_file in dicom_files:
        position = tuple(dicom_file.ImagePositionPatient)
        if position in positions:
            positions[position].append(dicom_file)
        else:
            positions[position] = [dicom_file]
    sorted_positions = sorted(positions.items(), key=lambda x: x[0][2])
    return [files for position, files in sorted_positions]
