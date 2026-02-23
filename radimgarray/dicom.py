""" Module for handling dicom files.
Holds all functions to load, save and check dicom files.


Dicom info (dict):
{
    "type": "dicom",
    "path": (Path),
    "header": [{},...]
    "affine": np.eye(4), not implemented jet
    "shape": (x, y, z, t)
}
"""

from __future__ import annotations
import numpy as np
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, MRImageStorage, UID
from pydicom.dataset import FileMetaDataset
from pydicom.errors import InvalidDicomError
from pathlib import Path
import warnings
from datetime import datetime
import nibabel.nicom.dicomwrappers as dicomwrappers


class DicomImage:
    def __init__(self):
        self.header = (
            None  # list of dict with header information for each loaded dicom file
        )
        # Note: affine matrix calculation is implemented in _calculate_affine_from_dicom()
        self.affine = np.eye(
            4
        )  # affine matrix for position of image array data in reference space
        self.shape = None


def load(path: Path) -> tuple[np.ndarray | None, dict] | None:
    """Load dicom files from a directory.

    If multiple series are discovered in the directory, the user is asked to select one.
    Args:
        path (Path): folder containing dicom files or folders with files

    Returns:
        dicom_matrix (np.ndarray): containing dicom data
        info (dict): containing additional information about the dicom data
            dicom: Module for handling dicom files.
    """
    dicom_data = []
    for file in path.glob("**/*"):
        if file.suffix == ".dcm" or file.suffix == ".dicom" or file.suffix == "":
            if file.is_file():
                try:
                    dicom_data.append(pydicom.dcmread(file))
                except InvalidDicomError:
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
        # dicom_matrix = np.permute_dims(dicom_matrix, [1, 2, 0])
        dicom_matrix = np.transpose(dicom_matrix, (1, 2, 0))
    if dicom_matrix.ndim == 4:
        dicom_matrix = np.transpose(dicom_matrix, (2, 3, 1, 0))
        # dicom_matrix = np.permute_dims(dicom_matrix, [2, 3, 1, 0])
    
    # Calculate affine from DICOM headers
    affine = _calculate_affine_from_dicom(dicom_series_sorted)
    
    info = {
        "type": "dicom",
        "path": path,
        "header": [dcm[0] for dcm in dicom_series_sorted],  # Store the first DICOM dataset for each position
        "affine": affine,
        "shape": dicom_matrix.shape,
    }
    return dicom_matrix, info


def save(array: np.ndarray, path: Path, info: dict) -> Path:
    """Save array data as multi-file DICOM series.

    Creates a timestamp-based subfolder containing one DICOM file per slice/timepoint.
    Handles three modes: DICOM source (preserve metadata), NIfTI source (use affine),
    or array source (minimal headers).

    Args:
        array: Data array to save (y, x, slices) or (y, x, slices, time)
        path: Target directory path
        info: Metadata dictionary containing type, header, affine, shape

    Returns:
        Path to the created series subfolder
    """
    # Convert array to DICOM-compatible format
    converted_array, scaling_info = _convert_array_to_dicom_compatible(array)
    
    # Create timestamp-based subfolder
    subfolder_name = _generate_timestamp_folder_name()
    output_dir = Path(path) / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique SeriesInstanceUID for this export
    series_uid = _generate_dicom_uid()
    
    # Reverse the transpose applied during loading
    # Loading does: 3D: (slices, y, x) -> (y, x, slices)
    #               4D: (slices, time, y, x) -> (y, x, slices, time)
    # So we need to reverse: 3D: (y, x, slices) -> (slices, y, x)
    #                        4D: (y, x, slices, time) -> (slices, time, y, x)
    
    if converted_array.ndim == 3:
        # 3D: transpose (y, x, slices) -> (slices, y, x)
        dicom_order = np.transpose(converted_array, (2, 0, 1))
        n_slices = dicom_order.shape[0]
        n_timepoints = 1
    elif converted_array.ndim == 4:
        # 4D: transpose (y, x, slices, time) -> (slices, time, y, x)
        dicom_order = np.transpose(converted_array, (2, 3, 0, 1))
        n_slices = dicom_order.shape[0]
        n_timepoints = dicom_order.shape[1]
    else:
        raise ValueError(f"Expected 3D or 4D array, got {converted_array.ndim}D")
    
    # Determine how to handle headers
    source_type = info.get("type", "array")
    has_dicom_headers = source_type == "dicom" and "header" in info and info["header"]
    has_affine = "affine" in info and info["affine"] is not None
    
    # Get original filenames if available (from DICOM source)
    original_path = info.get("path", None)
    original_filenames = []
    if has_dicom_headers and original_path and original_path.exists():
        # Try to get original filenames
        for file in sorted(original_path.glob("**/*.dcm")):
            original_filenames.append(file.stem)
        if not original_filenames:
            for file in sorted(original_path.glob("**/*.dicom")):
                original_filenames.append(file.stem)
    
    # Generate geometry tags from affine if available
    geometry_tags_list = []
    if has_affine:
        for slice_idx in range(n_slices):
            geometry_tags = _affine_to_dicom_tags(info["affine"], slice_idx)
            geometry_tags_list.append(geometry_tags)
    else:
        # No affine available - use defaults
        warnings.warn(
            "No affine matrix available. Using default geometry (identity orientation, unit spacing).",
            UserWarning,
        )
        default_geometry = {
            "ImageOrientationPatient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "PixelSpacing": [1.0, 1.0],
            "ImagePositionPatient": [0.0, 0.0, 0.0],
            "SliceThickness": 1.0,
        }
        geometry_tags_list = [default_geometry.copy() for _ in range(n_slices)]
        # Update Z position for each slice
        for slice_idx in range(n_slices):
            geometry_tags_list[slice_idx]["ImagePositionPatient"] = [0.0, 0.0, float(slice_idx)]
    
    # Counter for instance numbers (sequential across all files)
    instance_number = 1
    
    # Iterate through all slices and timepoints
    for slice_idx in range(n_slices):
        for time_idx in range(n_timepoints):
            # Get the 2D slice
            if n_timepoints == 1:
                slice_data = dicom_order[slice_idx, :, :]
            else:
                slice_data = dicom_order[slice_idx, time_idx, :, :]
            
            # Determine output filename
            if original_filenames and instance_number <= len(original_filenames):
                # Use original filename as basis
                base_name = original_filenames[instance_number - 1]
                filename = f"{base_name}_{instance_number:04d}.dcm"
            else:
                # Use generic naming
                filename = f"IM-{instance_number:04d}.dcm"
            
            output_path = output_dir / filename
            
            # Get geometry for this slice
            geometry_tags = geometry_tags_list[slice_idx]
            
            # Create or copy DICOM dataset
            if has_dicom_headers:
                # Mode 1: Copy from DICOM source
                # Get the corresponding header (DICOM Dataset object)
                header_idx = min(slice_idx, len(info["header"]) - 1)
                original_ds = info["header"][header_idx]
                
                # Create a copy of the original dataset
                ds = pydicom.Dataset(original_ds)
                
                # Update instance-specific tags
                ds.SOPInstanceUID = _generate_dicom_uid()
                ds.InstanceNumber = instance_number
                ds.SeriesInstanceUID = series_uid
                
                # Update geometry tags from affine if available
                if has_affine:
                    ds.ImageOrientationPatient = geometry_tags["ImageOrientationPatient"]
                    ds.ImagePositionPatient = geometry_tags["ImagePositionPatient"]
                    ds.PixelSpacing = geometry_tags["PixelSpacing"]
                    ds.SliceThickness = geometry_tags["SliceThickness"]
                
                # Update temporal tags for 4D
                if n_timepoints > 1:
                    ds.TemporalPositionIdentifier = str(time_idx + 1)
                    ds.TemporalPositionIndex = time_idx + 1
                    # Update acquisition time with offset
                    if hasattr(ds, "AcquisitionTime"):
                        try:
                            base_time = datetime.strptime(ds.AcquisitionTime, "%H%M%S")
                            offset_time = base_time.replace(second=(base_time.second + time_idx) % 60)
                            ds.AcquisitionTime = offset_time.strftime("%H%M%S")
                        except:
                            ds.AcquisitionTime = datetime.now().strftime("%H%M%S")
                
                # Update pixel data type tags
                ds.BitsAllocated = scaling_info["bits_allocated"]
                ds.BitsStored = scaling_info["bits_stored"]
                ds.HighBit = scaling_info["bits_stored"] - 1
                ds.PixelRepresentation = scaling_info["pixel_representation"]
                ds.Rows = slice_data.shape[0]
                ds.Columns = slice_data.shape[1]
                
                # Update rescale parameters
                if scaling_info["rescale_slope"] != 1.0 or scaling_info["rescale_intercept"] != 0.0:
                    ds.RescaleSlope = scaling_info["rescale_slope"]
                    ds.RescaleIntercept = scaling_info["rescale_intercept"]
                
                # Ensure file meta is present
                if not hasattr(ds, "file_meta") or ds.file_meta is None:
                    file_meta = FileMetaDataset()
                    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    file_meta.MediaStorageSOPClassUID = getattr(ds, "SOPClassUID", MRImageStorage)
                    file_meta.MediaStorageSOPInstanceUID = UID(ds.SOPInstanceUID)
                    file_meta.ImplementationClassUID = UID(_generate_dicom_uid())
                    ds.file_meta = file_meta
                
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                
            else:
                # Mode 2 & 3: Create minimal header (NIfTI source or array source)
                if source_type == "nifti":
                    # NIfTI source: geometry from affine is already in geometry_tags_list
                    pass
                else:
                    # Array source: using default geometry with warning already issued
                    pass
                
                ds = _create_minimal_dicom_header(
                    array_shape=slice_data.shape,
                    slice_index=slice_idx,
                    time_index=time_idx,
                    instance_number=instance_number,
                    series_uid=series_uid,
                    geometry_tags=geometry_tags,
                    scaling_info=scaling_info,
                )
            
            # Set pixel data
            ds.PixelData = slice_data.tobytes()
            
            # Save DICOM file
            ds.save_as(output_path, write_like_original=False)
            
            instance_number += 1
    
    print(f"Saved {instance_number - 1} DICOM files to {output_dir}")
    return output_dir


def get_series_data(series_list: list, interface: str = "cli") -> list:
    """Get series data from a list of dicom files.

    Args:
        series_list (list): of all found dicom files
        interface (str): select interface for user interaction when multiple dicom
            series are found
    Returns:
        single_series_list (list): of dicom data from a single series
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
                f"Enter the number of the series you want to load [0-{len(unique_series_ids) - 1}]: "
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
    """Sort dicom files by ImagePosition and 4. dimension functional data.

    From IntervalLudger - GoNifti
    Args:
        dicom_files (list): unsorted list of dicom files
    Returns:
        (list): list containing list of sorted dicom files
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


def _calculate_affine_from_dicom(
    dicom_datasets: list[list[pydicom.Dataset]],
) -> np.ndarray:
    """Calculate affine transformation matrix from DICOM headers using nibabel.

    Uses nibabel's dicomwrappers to compute the affine matrix in LPS coordinate system.
    Falls back to identity matrix with warnings if geometry tags are missing.

    Args:
        dicom_datasets: List of lists of sorted DICOM datasets [spatial_position][temporal_frames]

    Returns:
        4x4 affine transformation matrix (LPS coordinate system)
    """
    try:
        # Get first DICOM file for geometry calculation
        ref_dcm = dicom_datasets[0][0]

        # Check for required tags
        required_tags = ["ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"]
        missing_tags = [tag for tag in required_tags if not hasattr(ref_dcm, tag)]

        if missing_tags:
            warnings.warn(
                f"Missing required DICOM tags for affine calculation: {missing_tags}. "
                "Using identity matrix with default spacing.",
                UserWarning,
            )
            # Create default affine with unit spacing
            affine = np.eye(4)
            if hasattr(ref_dcm, "PixelSpacing"):
                pixel_spacing = ref_dcm.PixelSpacing
                affine[0, 0] = pixel_spacing[1]
                affine[1, 1] = pixel_spacing[0]
            return affine

        # Use nibabel's dicomwrappers for robust affine calculation
        # Create a temporary wrapper from the dataset
        wrapper = dicomwrappers.Wrapper(ref_dcm)
        
        # Get affine from wrapper (handles LPS coordinate system)
        affine = wrapper.affine
        
        # If we have multiple slices, verify slice spacing
        if len(dicom_datasets) > 1:
            pos_1 = np.array(dicom_datasets[0][0].ImagePositionPatient)
            pos_2 = np.array(dicom_datasets[1][0].ImagePositionPatient)
            actual_spacing = np.linalg.norm(pos_2 - pos_1)
            
            # Update slice spacing in affine if different
            slice_vector = affine[:3, 2]
            current_spacing = np.linalg.norm(slice_vector)
            if not np.isclose(current_spacing, actual_spacing, rtol=0.01):
                affine[:3, 2] = (slice_vector / current_spacing) * actual_spacing

        return affine

    except Exception as e:
        warnings.warn(
            f"Failed to calculate affine from DICOM headers: {e}. "
            "Using identity matrix.",
            UserWarning,
        )
        return np.eye(4)


def _convert_array_to_dicom_compatible(
    array: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Convert array to DICOM-compatible data type with optimal scaling.

    Converts float arrays to int16 with optimal RescaleSlope/RescaleIntercept
    to minimize precision loss. Preserves int16/uint16 arrays unchanged.

    Args:
        array: Input array of any dtype

    Returns:
        Tuple of (converted_array, scaling_info_dict)
        scaling_info contains: dtype, bits_allocated, bits_stored, pixel_representation,
                               rescale_slope, rescale_intercept
    """
    original_dtype = array.dtype
    
    # Default scaling parameters
    scaling_info = {
        "dtype": original_dtype,
        "bits_allocated": 16,
        "bits_stored": 16,
        "pixel_representation": 1,  # signed
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
    }

    # Handle different data types
    if original_dtype in [np.int16]:
        # Already DICOM compatible (signed)
        scaling_info["pixel_representation"] = 1
        return array, scaling_info
    
    elif original_dtype in [np.uint16]:
        # Already DICOM compatible (unsigned)
        scaling_info["pixel_representation"] = 0
        return array, scaling_info
    
    elif original_dtype in [np.float32, np.float64, np.float16]:
        # Convert float to int16 with optimal scaling
        warnings.warn(
            f"Converting {original_dtype} to int16 for DICOM compatibility. "
            "Using optimal RescaleSlope/RescaleIntercept to minimize precision loss.",
            UserWarning,
        )
        
        # Calculate optimal scaling parameters
        array_min = float(np.nanmin(array))
        array_max = float(np.nanmax(array))
        
        # Target range for int16: -32768 to 32767
        int16_min = -32768
        int16_max = 32767
        
        if array_max == array_min:
            # Constant array
            rescale_slope = 1.0
            rescale_intercept = array_min
            converted_array = np.zeros_like(array, dtype=np.int16)
        else:
            # Calculate optimal slope and intercept
            # Formula: original_value = rescale_slope * stored_value + rescale_intercept
            rescale_slope = (array_max - array_min) / (int16_max - int16_min)
            rescale_intercept = array_min - (rescale_slope * int16_min)
            
            # Convert array
            converted_array = np.round((array - rescale_intercept) / rescale_slope).astype(np.int16)
        
        scaling_info["rescale_slope"] = rescale_slope
        scaling_info["rescale_intercept"] = rescale_intercept
        scaling_info["pixel_representation"] = 1  # signed int16
        scaling_info["dtype"] = np.int16
        
        return converted_array, scaling_info
    
    elif original_dtype in [np.uint32, np.uint64]:
        # Convert to uint16 with scaling
        warnings.warn(
            f"Converting {original_dtype} to uint16 for DICOM compatibility. "
            "Data may lose precision if values exceed uint16 range.",
            UserWarning,
        )
        
        array_min = float(np.min(array))
        array_max = float(np.max(array))
        
        if array_max <= 65535:
            # Fits in uint16 without scaling
            converted_array = array.astype(np.uint16)
        else:
            # Need scaling
            rescale_slope = array_max / 65535.0
            rescale_intercept = 0.0
            converted_array = np.round(array / rescale_slope).astype(np.uint16)
            scaling_info["rescale_slope"] = rescale_slope
            scaling_info["rescale_intercept"] = rescale_intercept
        
        scaling_info["pixel_representation"] = 0  # unsigned
        scaling_info["dtype"] = np.uint16
        
        return converted_array, scaling_info
    
    elif original_dtype in [np.int32, np.int64, np.int8]:
        # Convert to int16 with scaling if needed
        warnings.warn(
            f"Converting {original_dtype} to int16 for DICOM compatibility.",
            UserWarning,
        )
        
        array_min = float(np.min(array))
        array_max = float(np.max(array))
        
        if array_min >= -32768 and array_max <= 32767:
            # Fits in int16 without scaling
            converted_array = array.astype(np.int16)
        else:
            # Need scaling
            rescale_slope = (array_max - array_min) / 65535.0
            rescale_intercept = array_min
            converted_array = np.round((array - rescale_intercept) / rescale_slope).astype(np.int16)
            scaling_info["rescale_slope"] = rescale_slope
            scaling_info["rescale_intercept"] = rescale_intercept
        
        scaling_info["pixel_representation"] = 1
        scaling_info["dtype"] = np.int16
        
        return converted_array, scaling_info
    
    else:
        # Fallback for other types
        warnings.warn(
            f"Unsupported dtype {original_dtype}. Converting to int16.",
            UserWarning,
        )
        converted_array = array.astype(np.int16)
        return converted_array, scaling_info


def _affine_to_dicom_tags(
    affine: np.ndarray, slice_index: int = 0
) -> dict:
    """Decompose affine matrix into DICOM geometry tags.

    Extracts ImageOrientationPatient, PixelSpacing, ImagePositionPatient,
    and SliceThickness from a 4x4 affine transformation matrix.

    Args:
        affine: 4x4 affine transformation matrix (LPS coordinate system)
        slice_index: Index of the current slice for position calculation

    Returns:
        Dictionary with DICOM geometry tags
    """
    # Extract column and row direction vectors with spacing
    col_vector = affine[:3, 0]
    row_vector = affine[:3, 1]
    slice_vector = affine[:3, 2]
    
    # Calculate spacing (magnitude of vectors)
    col_spacing = np.linalg.norm(col_vector)
    row_spacing = np.linalg.norm(row_vector)
    slice_spacing = np.linalg.norm(slice_vector)
    
    # Get unit direction vectors
    col_direction = col_vector / col_spacing if col_spacing > 0 else col_vector
    row_direction = row_vector / row_spacing if row_spacing > 0 else row_vector
    
    # Calculate position for this slice
    origin = affine[:3, 3]
    slice_offset = slice_vector * slice_index
    position = origin + slice_offset
    
    # Build DICOM tags dictionary
    tags = {
        "ImageOrientationPatient": [
            float(col_direction[0]),
            float(col_direction[1]),
            float(col_direction[2]),
            float(row_direction[0]),
            float(row_direction[1]),
            float(row_direction[2]),
        ],
        "PixelSpacing": [float(row_spacing), float(col_spacing)],
        "ImagePositionPatient": [float(position[0]), float(position[1]), float(position[2])],
        "SliceThickness": float(slice_spacing),
    }
    
    return tags


def _generate_timestamp_folder_name() -> str:
    """Generate timestamp-based folder name for DICOM series.

    Returns:
        Folder name string like 'series_20260128_143022'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"series_{timestamp}"


def _generate_dicom_uid() -> str:
    """Generate a unique DICOM UID.

    Returns:
        Unique DICOM UID string
    """
    return generate_uid()


def _create_minimal_dicom_header(
    array_shape: tuple,
    slice_index: int,
    time_index: int,
    instance_number: int,
    series_uid: str,
    geometry_tags: dict,
    scaling_info: dict,
) -> pydicom.Dataset:
    """Create a minimal valid DICOM header.

    Args:
        array_shape: Shape of the 2D slice (rows, cols)
        slice_index: Spatial slice index
        time_index: Temporal frame index
        instance_number: DICOM InstanceNumber
        series_uid: SeriesInstanceUID
        geometry_tags: Dictionary with ImageOrientationPatient, PixelSpacing, ImagePositionPatient
        scaling_info: Dictionary with pixel data type information

    Returns:
        pydicom.Dataset with minimal required tags
    """
    ds = pydicom.Dataset()
    
    # Patient Module (Type 2 - required, can be empty)
    ds.PatientName = "ANONYMIZED"
    ds.PatientID = "ANON"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    
    # General Study Module
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyInstanceUID = _generate_dicom_uid()
    ds.StudyID = "1"
    ds.AccessionNumber = ""
    
    # General Series Module
    ds.SeriesDate = datetime.now().strftime("%Y%m%d")
    ds.SeriesTime = datetime.now().strftime("%H%M%S")
    ds.Modality = "OT"  # Other
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "Exported from RadImgArray"
    
    # General Equipment Module
    ds.Manufacturer = "RadImgArray"
    ds.ManufacturerModelName = "Python Export"
    
    # General Image Module
    ds.InstanceNumber = instance_number
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.AcquisitionDate = datetime.now().strftime("%Y%m%d")
    ds.AcquisitionTime = datetime.now().strftime("%H%M%S")
    
    # Add temporal position if 4D
    if time_index > 0:
        ds.TemporalPositionIdentifier = str(time_index + 1)
        ds.TemporalPositionIndex = time_index + 1
        # Add acquisition time offset (assuming 1 second TR)
        base_time = datetime.now()
        offset_seconds = time_index
        ds.AcquisitionTime = (base_time.replace(second=(base_time.second + offset_seconds) % 60)).strftime("%H%M%S")
    
    # Image Plane Module
    if geometry_tags:
        ds.ImagePositionPatient = geometry_tags["ImagePositionPatient"]
        ds.ImageOrientationPatient = geometry_tags["ImageOrientationPatient"]
        ds.SliceThickness = geometry_tags["SliceThickness"]
        ds.PixelSpacing = geometry_tags["PixelSpacing"]
    
    # Image Pixel Module
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = array_shape[0]
    ds.Columns = array_shape[1]
    ds.BitsAllocated = scaling_info["bits_allocated"]
    ds.BitsStored = scaling_info["bits_stored"]
    ds.HighBit = scaling_info["bits_stored"] - 1
    ds.PixelRepresentation = scaling_info["pixel_representation"]
    
    # Rescale parameters
    if scaling_info["rescale_slope"] != 1.0 or scaling_info["rescale_intercept"] != 0.0:
        ds.RescaleSlope = scaling_info["rescale_slope"]
        ds.RescaleIntercept = scaling_info["rescale_intercept"]
        ds.RescaleType = "US"  # Unspecified
    
    # SOP Common Module
    ds.SOPClassUID = MRImageStorage  # Using MR Image Storage as default
    ds.SOPInstanceUID = _generate_dicom_uid()
    
    # File Meta Information
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = UID(ds.SOPInstanceUID)
    file_meta.ImplementationClassUID = UID(_generate_dicom_uid())
    
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    return ds
