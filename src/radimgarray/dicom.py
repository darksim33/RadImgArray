import numpy as np
import pydicom
import argparse
from pathlib import Path

class DicomImage:
    def __init__(self):
        self.data = None
        self.header = None


def load(path: Path):
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
        dicom_matrix.append([dcm[idx].pixel_array for dcm in dicom_series_sorted])

    dicom_matrix = np.array(dicom_matrix)
    if dicom_matrix.ndim == 3:
        np.permute_dims(dicom_matrix, [1, 2, 0])
    if dicom_matrix.ndim == 4:
        np.permute_dims(dicom_matrix, [2, 3, 1, 0])
    dicom_header = [dcm[0].items for dcm in dicom_series_sorted]
    return dicom_matrix, dicom_header



def get_series_data(series_list: list, interface: str = "cli"):
    series_ids = []
    series_to_erase = []
    for series in series_list:
        try:
            uid = series.SeriesInstanceUID
            series_ids.append(uid)
        except AttributeError:
            series_to_erase.append(series)

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
                print(f"[{idx}]: {series_list[series_ids.index(series_id)].SeriesDescription}")
            number = input(f"Enter the number of the series you want to load [0-{len(unique_series_ids)-1}]: ")
            print(f"Loading series number {number}")
            series_id = unique_series_ids[int(number)]
        else:
            # Placeholder for GUI
            series_id = unique_series_ids[0]

    else:
        series_id = unique_series_ids[0]

    series_indexes = [idx for idx, x in enumerate(series_ids) if series_id == x]
    return series_list[series_indexes[0]:series_indexes[-1]+1]

def sort_dicom_files(dicom_files: list[pydicom.Dataset]) -> list:
    """From Ludgers GoNifti"""
    positions = {}
    for dicom_file in dicom_files:
        position = tuple(dicom_file.ImagePositionPatient)
        if position in positions:
            positions[position].append(dicom_file)
        else:
            positions[position] = [dicom_file]
    sorted_positions = sorted(positions.items(), key=lambda x: x[0][2])
    return [files for position, files in sorted_positions]