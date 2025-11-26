# Video Editing Tools

A Python toolkit for video processing and editing using OpenCV and FFmpeg. This package provides tools for selecting regions of interest (ROI) from videos and creating processed versions with spatial cropping, downsampling, frame skipping, and other transformations.

## Setup

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/) package manager
- Python 3.8 or higher

### Environment Setup

1. **Create the environment from the YAML file:**

   ```bash
   mamba env create -f environment.yml
   ```

   Or if you're using conda:

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**

   ```bash
   mamba activate video_edits
   ```

   Or with conda:

   ```bash
   conda activate video_edits
   ```

3. **Verify installation:**

   ```bash
   python -c "import cv2, subprocess; subprocess.run(['ffmpeg', '-version'])"
   ```

## Usage

### Quick Start

The easiest way to get started is to use the provided `main.py` file as a template. Open it in a Jupyter notebook or Python script and modify the parameters to suit your needs.

### Main Functions

#### 1. Select Region of Interest (ROI)

The `select_roi()` function allows you to interactively select a rectangular region from the first frame of a video.

```python
import video_transforms as vt
import os

video_dir_path = "/path/to/video/directory"
video_filename = "your_video.mp4"

# Interactively select ROI
roi = vt.select_roi(video_dir_path, video_filename)
# Returns: (x, y, width, height) or None if cancelled
```

**How it works:**
- Displays the first frame of the video
- Click and drag to select a rectangular region
- Press **RETURN** to confirm the selection
- Press **ESC** to cancel
- Returns the ROI coordinates as `(x, y, width, height)` in original video coordinates

#### 2. Create Processed Video

The `create_small_video()` function processes videos with various transformations using FFmpeg.

```python
import video_transforms as vt

# Process a single video
output_path = vt.create_small_video(
    video_path="/path/to/video/directory",
    video_filename="input.mp4",
    roi=(100, 100, 500, 400),  # (x, y, width, height)
    downsample_factor=2,        # Downsample by 2x
    convert_to_gray=False,     # Keep color
    start_frame=0,             # Start from beginning
    end_frame=1000,            # End at frame 1000
    skip_frames=1,             # Keep all frames
    quality=80,                # Quality 0-100
    output_format=".mp4",      # Output format
    include_audio=False,        # Exclude audio
    enhance_contrast=False     # No contrast enhancement
)

# Process all videos in a directory
output_paths = vt.create_small_video(
    video_path="/path/to/video/directory",
    video_filename=None,  # None = process all videos
    roi=(100, 100, 500, 400),
    downsample_factor=2,
    quality=80
)
```

### Parameters

#### `create_small_video()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | str | required | Path to directory containing video(s) or full path to video file |
| `video_filename` | str/None | None | Name of video file. If None, processes all videos in directory |
| `roi` | tuple/None | None | Region of interest: `(x, y, width, height)` for spatial cropping |
| `downsample_factor` | int | 1 | Factor to downsample by (e.g., 2 = half size). Must be >= 1 |
| `convert_to_gray` | bool | False | Convert video to grayscale if True |
| `start_frame` | int | 0 | Starting frame number (0-indexed) |
| `end_frame` | int/None | None | Ending frame number (None for end of video) |
| `skip_frames` | int | 1 | Keep every Nth frame (e.g., 2 = keep every 2nd frame) |
| `output_dir_path` | str/None | None | Output directory (None = same as input directory) |
| `quality` | int | 80 | Quality level: 0 (max compression) to 100 (no compression) |
| `output_format` | str/None | None | Output format (e.g., '.mp4', '.mov'). None = same as input |
| `include_audio` | bool | False | Include audio in output if True |
| `enhance_contrast` | bool | False | Apply histogram equalization for contrast enhancement |

### Examples

#### Example 1: Basic Video Processing

```python
import video_transforms as vt
import os

# Setup paths
video_dir = "/Users/username/Videos"
video_file = "my_video.mp4"

# Process video: crop, downsample, trim
output = vt.create_small_video(
    video_path=video_dir,
    video_filename=video_file,
    roi=(100, 100, 800, 600),  # Crop to 800x600 region
    downsample_factor=2,        # Reduce to half size
    start_frame=100,            # Start at frame 100
    end_frame=2000,             # End at frame 2000
    quality=90                  # High quality
)
```

#### Example 2: Interactive ROI Selection

```python
import video_transforms as vt
import os

video_dir = "/Users/username/Videos"
video_file = "my_video.mp4"

# First, select the region of interest interactively
roi = vt.select_roi(video_dir, video_file)

if roi:
    print(f"Selected ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    # Then process the video with the selected ROI
    output = vt.create_small_video(
        video_path=video_dir,
        video_filename=video_file,
        roi=roi,
        downsample_factor=2,
        quality=80
    )
else:
    print("ROI selection cancelled")
```

#### Example 3: Batch Processing

```python
import video_transforms as vt

# Process all videos in a directory
# Note: Files ending in "_small" are automatically excluded
output_paths = vt.create_small_video(
    video_path="/path/to/videos",
    video_filename=None,  # None = process all videos
    roi=(463, 379, 493, 183),
    downsample_factor=2,
    convert_to_gray=False,
    start_frame=0,
    end_frame=300,
    skip_frames=1,
    quality=100,           # Lossless quality
    output_format=".mp4",
    include_audio=False,
    enhance_contrast=True  # Apply histogram equalization
)

print(f"Processed {len(output_paths)} videos")
```

#### Example 4: Grayscale with Contrast Enhancement

```python
import video_transforms as vt

output = vt.create_small_video(
    video_path="/path/to/video",
    video_filename="input.mp4",
    convert_to_gray=True,      # Convert to grayscale
    enhance_contrast=True,     # Apply histogram equalization
    downsample_factor=2,
    quality=80
)
```

### Output Files

- Output files are automatically named with `_small` suffix (e.g., `video.mp4` → `video_small.mp4`)
- Existing output files are automatically overwritten
- When processing multiple files, files ending in `_small` are automatically excluded from processing

### Progress Tracking

The function provides progress updates every 100 frames:

```
Processing video.mp4
   Processing frame 100/1000 (10.0%)
   Processing frame 200/1000 (20.0%)
   ...
   Video processing complete. Output saved to: /path/to/video_small.mp4
```

## Command Line Usage

You can also use the functions from the command line by creating a Python script:

```bash
# Activate environment
mamba activate video_edits

# Run your script
python your_script.py
```

Or use Python interactively:

```bash
python
>>> import video_transforms as vt
>>> roi = vt.select_roi("/path/to/videos", "video.mp4")
>>> vt.create_small_video("/path/to/videos", "video.mp4", roi=roi)
```

## Technical Details

### Video Codecs

- **MP4/M4V**: Uses H.264 codec for maximum QuickTime compatibility on macOS
- **MOV**: Uses H.264 codec
- **MKV**: Uses H.265 (HEVC) codec for better compression
- **WebM**: Uses VP9 codec
- **AVI**: Uses H.264 codec

### Quality Settings

- Quality 0-100 maps to CRF values (0 = lossless, 51 = maximum compression)
- Quality 100 uses lossless encoding (no profile constraints)
- Lower quality values provide better compression but larger file sizes

### Requirements

- Dimensions are automatically adjusted to be even (divisible by 2) for H.264 encoding
- Histogram equalization uses FFmpeg's `histeq` filter
- Frame skipping uses FFmpeg's `select` filter

## Troubleshooting

### Video won't play in QuickTime

- Ensure you're using MP4 format with H.264 codec (default)
- Try quality values less than 100 (lossless encoding may have compatibility issues)

### "height not divisible by 2" error

- This should be automatically handled, but if you encounter it, ensure your ROI dimensions are even numbers

### Files not overwriting

- The function automatically deletes existing output files before processing
- Ensure you have write permissions in the output directory

## License

See LICENSE file for details.
