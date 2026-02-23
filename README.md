# RadImgArray

A Python library for working with radiological imaging data, providing enhanced numpy arrays with built-in support for DICOM and NIfTI file formats.

## Features

- **Enhanced Array Classes**: `ImgArray` and `SegArray` extend numpy arrays with medical imaging metadata
- **DICOM Support**: Load and save multi-file DICOM series with metadata preservation
- **NIfTI Support**: Read and write NIfTI files (.nii, .nii.gz) with header information
- **Format Conversion**: Convert between DICOM and NIfTI formats
- **Segmentation Tools**: Specialized array class for segmentation masks with helper methods
- **Visualization**: Built-in plotting functions for 2D, 3D, and 4D medical images

## Installation

```bash
pip install radimgarray
```

Or with uv:

```bash
uv add radimgarray
```

## Quick Start

```python
from radimgarray import ImgArray, SegArray

# Load DICOM series
img = ImgArray("path/to/dicom/folder")

# Load NIfTI file
img = ImgArray("path/to/image.nii.gz")

# Access image data (fully compatible with numpy)
print(img.shape)
print(img.mean())

# Access metadata
print(img.info["affine"])
print(img.info["header"])

# Save to different format
img.save("output.nii.gz")  # Save as NIfTI
img.save("output_dicom/")  # Save as DICOM series

# Visualize
img.show()

# Work with segmentation masks
seg = SegArray("path/to/segmentation.nii.gz")
print(seg.seg_values)  # Get unique segment values
print(seg.number_segs)  # Count segments
indices = seg.get_seg_indices(1)  # Get indices for segment 1
```

## API Overview

### ImgArray

Enhanced numpy array for radiological images with metadata support:

- `ImgArray(input)`: Create from file path, numpy array, or list
- `.info`: Dictionary containing metadata (type, path, header, affine, shape)
- `.save(path, save_as=None)`: Save to DICOM or NIfTI format
- `.show()`: Display the image

### SegArray

Specialized array for segmentation masks:

- Inherits all `ImgArray` functionality
- `.seg_values`: List of unique segmentation values (excluding background)
- `.number_segs`: Number of segments
- `.get_seg_indices(value)`: Get indices for a specific segment value

### Tools Module

Utility functions for image processing:

- `zero_pad_to_square()`: Pad images to square dimensions
- `get_mean_signal()`: Calculate mean signal in segmented regions
- `get_single_seg_array()`: Extract a single segment
- `mean_seg_signals_to_excel()`: Export signal analysis to Excel
- Array visualization helpers

## Requirements

- Python 3.9+
- numpy
- nibabel
- pydicom
- matplotlib
- imantics
- pandas
- openpyxl

## Development

```bash
# Clone the repository
git clone https://github.com/darksim33/RadImgArray.git
cd RadImgArray

# Install with uv
uv sync --all-groups

# Run tests
uv run pytest tests
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
