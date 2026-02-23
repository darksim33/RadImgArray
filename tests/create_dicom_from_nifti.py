"""Utility script to convert NIfTI test data to DICOM for testing."""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import radimgarray as ria


def convert_nifti_to_dicom():
    """Convert the test NIfTI file to DICOM test series."""
    # Path to NIfTI test file
    nifti_path = Path(__file__).parent / ".assets" / "test_img.nii.gz"
    
    if not nifti_path.exists():
        print(f"Error: NIfTI file not found at {nifti_path}")
        return
    
    print(f"Loading NIfTI file: {nifti_path}")
    
    # Load NIfTI data
    rad_img = ria.ImgArray(nifti_path)
    
    print(f"Loaded array shape: {rad_img.shape}")
    print(f"Array dtype: {rad_img.dtype}")
    print(f"Affine matrix:\n{rad_img.info['affine']}")
    
    # Create output directory
    output_dir = Path(__file__).parent / ".temp" / "dicom_from_nifti"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving as DICOM to: {output_dir}")
    
    # Save as DICOM
    result_path = rad_img.save(output_dir, file_type="dicom")
    
    print(f"DICOM series saved to: {result_path}")
    
    # Check if result_path is valid
    if result_path is None or not Path(result_path).exists():
        print("Warning: save() returned None or invalid path")
        # Try to find the created series folder
        created_folders = list(output_dir.glob("series_*"))
        if created_folders:
            result_path = created_folders[0]
            print(f"Found created series at: {result_path}")
        else:
            print("Error: No series folder found")
            return
    
    # Verify by loading back
    print("\n--- Verification: Loading DICOM back ---")
    dicom_img = ria.ImgArray(result_path)
    
    print(f"Reloaded shape: {dicom_img.shape}")
    print(f"Reloaded dtype: {dicom_img.dtype}")
    print(f"Affine matrix:\n{dicom_img.info['affine']}")
    
    # Check if data matches (accounting for dtype conversion)
    import numpy as np
    
    # The data may have been converted to int16, so we need to account for rescaling
    if 'header' in dicom_img.info and dicom_img.info['header']:
        print("\n--- Checking data preservation ---")
        # Data will be scaled differently, so just check shape and rough range
        print(f"Original range: [{rad_img.min():.2f}, {rad_img.max():.2f}]")
        print(f"Reloaded range: [{dicom_img.min():.2f}, {dicom_img.max():.2f}]")
        print(f"Shape match: {rad_img.shape == dicom_img.shape}")
    
    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    convert_nifti_to_dicom()
