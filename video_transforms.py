import cv2
import os
import subprocess
import re


# Global variables for ROI selection
_roi_selection = {
    'selecting': False,
    'start_point': None,
    'end_point': None,
    'roi': None,
    'done': False
}


def _mouse_callback(event, x, y, flags, param):
    """Mouse callback for ROI selection."""
    frame_copy = param['frame'].copy()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_selection['selecting'] = True
        _roi_selection['start_point'] = (x, y)
        _roi_selection['end_point'] = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and _roi_selection['selecting']:
        _roi_selection['end_point'] = (x, y)
        # Draw rectangle as user drags
        cv2.rectangle(frame_copy, _roi_selection['start_point'], _roi_selection['end_point'], (0, 255, 0), 2)
        cv2.imshow("Select ROI - Click and drag, then press RETURN to confirm or ESC to cancel", frame_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:
        _roi_selection['selecting'] = False
        _roi_selection['end_point'] = (x, y)
        # Draw final rectangle
        cv2.rectangle(frame_copy, _roi_selection['start_point'], _roi_selection['end_point'], (0, 255, 0), 2)
        cv2.imshow("Select ROI - Click and drag, then press RETURN to confirm or ESC to cancel", frame_copy)


def select_roi(video_path, video_filename, max_display_width=1920, max_display_height=1080):
    """
    Select a region of interest from a video file.
    
    Args:
        video_path: Path to the directory containing the video file
        video_filename: Name of the video file
        max_display_width: Maximum width for display (default: 1920)
        max_display_height: Maximum height for display (default: 1080)
        
    Returns:
        tuple: (x, y, w, h) coordinates of the selected ROI in original frame coordinates, or None if cancelled
    """
    # Reset global state
    _roi_selection['selecting'] = False
    _roi_selection['start_point'] = None
    _roi_selection['end_point'] = None
    _roi_selection['roi'] = None
    _roi_selection['done'] = False
    
    video_path = os.path.join(video_path, video_filename)
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set to read the first frame explicitly
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ret, frame = video.read()
    video.release()
    
    if not ret:
        raise ValueError(f"Could not read first frame from video: {video_path}")
    
    # Store original dimensions
    orig_height, orig_width = frame.shape[:2]
    
    # Resize frame for display if it's too large (improves performance)
    scale = 1.0
    if orig_width > max_display_width or orig_height > max_display_height:
        scale = min(max_display_width / orig_width, max_display_height / orig_height)
        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)
        display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_frame = frame
    
    # Create window and set mouse callback
    window_name = "Select ROI - Click and drag, then press RETURN to confirm or ESC to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback, {'frame': display_frame})
    
    # Display initial frame
    cv2.imshow(window_name, display_frame)
    
    result = None
    try:
        # Wait for user to select ROI
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # ESC key to cancel
            if key == 27:  # ESC
                result = None
                break
            
            # RETURN/ENTER key to confirm selection
            if key == 13:  # RETURN/ENTER
                if _roi_selection['start_point'] and _roi_selection['end_point']:
                    x1, y1 = _roi_selection['start_point']
                    x2, y2 = _roi_selection['end_point']
                    
                    # Ensure x1 < x2 and y1 < y2
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    # Scale ROI coordinates back to original frame size
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                    
                    result = (x, y, w, h)
                    print(f"Selected ROI coordinates: x={x}, y={y}, w={w}, h={h}")
                    break
    finally:
        # Ensure window is properly closed on macOS
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Process window close event
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Additional event processing for macOS
    
    return result


def _process_single_video(input_video, output_video, roi=None, downsample_factor=1, convert_to_gray=False, start_frame=0, end_frame=None, skip_frames=1, quality=80, include_audio=True, enhance_contrast=False):
    """
    Helper function to process a single video file with ffmpeg.
    
    Args:
        input_video: Full path to input video file
        output_video: Full path to output video file
        roi: Tuple (x, y, w, h) for spatial cropping, or None for no cropping
        downsample_factor: Factor to downsample by (e.g., 2 means half size). Must be >= 1
        convert_to_gray: If True, convert video to grayscale
        start_frame: Starting frame number (0-indexed)
        end_frame: Ending frame number (None for end of video)
        skip_frames: Keep every Nth frame (e.g., 2 means keep every 2nd frame). Must be >= 1
        quality: Quality level from 0-100 (100 = no compression, 0 = maximum compression)
        include_audio: If False, exclude audio from output. Default: True
        enhance_contrast: If True, apply histogram equalization to expand contrast to full range. Default: False
        
    Returns:
        str: Path to the output video file
    """
    if not os.path.exists(input_video):
        raise ValueError(f"Video file not found: {input_video}")
    
    # Delete output file if it exists to ensure clean overwrite
    if os.path.exists(output_video):
        try:
            os.remove(output_video)
        except OSError as e:
            raise RuntimeError(f"Could not delete existing output file {output_video}: {e}")
    
    # Get video properties for time calculations
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {input_video}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    
    if fps == 0:
        raise ValueError(f"Could not determine FPS from video: {input_video}")
    
    # Calculate expected output frames for progress reporting
    if end_frame is not None:
        # If end_frame is specified, calculate expected output frames
        input_frame_range = end_frame - start_frame
        expected_output_frames = int(input_frame_range / skip_frames) if skip_frames > 1 else input_frame_range
    else:
        # If processing entire video, account for start_frame and skip_frames
        input_frame_range = total_frames - start_frame
        expected_output_frames = int(input_frame_range / skip_frames) if skip_frames > 1 else input_frame_range
    
    # Build ffmpeg filter chain
    filters = []
    
    # Spatial cropping
    if roi is not None:
        x, y, w, h = roi
        filters.append(f"crop={w}:{h}:{x}:{y}")
    
    # Downsampling
    if downsample_factor > 1:
        filters.append(f"scale=iw/{downsample_factor}:ih/{downsample_factor}")
    
    # Grayscale conversion
    if convert_to_gray:
        filters.append("format=gray")
    
    # Contrast enhancement using histogram equalization
    if enhance_contrast:
        # Use histogram equalization (like cv2 CLAHE) which expands histogram to full range
        # This provides the type of contrast enhancement that expands the histogram to full range
        filters.append("histeq")
    
    # Frame skipping (using select filter)
    if skip_frames > 1:
        filters.append(f"select='not(mod(n\\,{skip_frames}))'")
    
    # Ensure dimensions are even for H.264 encoding (required for yuv420p)
    # This ensures width and height are divisible by 2
    filters.append("scale='trunc(iw/2)*2':'trunc(ih/2)*2'")
    
    # Combine filters
    filter_complex = ",".join(filters) if filters else None
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-stats_period", "0.1"]  # -y to overwrite output file, -stats_period for frequent updates
    
    # Time trimming: calculate start time and duration
    if start_frame > 0:
        start_time = start_frame / fps
        cmd.extend(["-ss", f"{start_time:.6f}"])
    
    # Input file
    cmd.extend(["-i", input_video])
    
    # Calculate duration if end_frame is specified
    if end_frame is not None:
        if start_frame > 0:
            duration_frames = end_frame - start_frame
        else:
            duration_frames = end_frame
        duration = duration_frames / fps
        cmd.extend(["-t", f"{duration:.6f}"])
    
    # Apply filters
    if filter_complex:
        cmd.extend(["-vf", filter_complex])
    
    # Determine codec and quality settings based on output format
    output_ext = os.path.splitext(output_video)[1].lower()
    
    # Map quality (0-100) to CRF values
    # Quality 100 -> CRF 0 (lossless/near-lossless), Quality 0 -> CRF 51 (maximum compression)
    # For H.264/H.265: CRF 0-51, where lower is better quality
    # For VP9: CRF 0-63, where lower is better quality
    
    # Choose codec based on output format - use H.264 for MP4 for QuickTime compatibility
    if output_ext in ['.mp4', '.m4v']:
        # H.264 for MP4/M4V (better QuickTime compatibility on Mac)
        # CRF scale: 0-51, map quality 100->0, 0->51
        crf_value = int(51 * (1 - quality / 100))
        crf_value = max(0, min(51, crf_value))
        # Use slower preset for better compression efficiency
        # movflags faststart allows playback to begin before file is fully downloaded
        cmd.extend(["-c:v", "libx264", "-preset", "slow", "-crf", str(crf_value), 
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        # Add profile and level for compatibility, but not for lossless (CRF=0)
        # High profile doesn't support lossless encoding
        if crf_value > 0:
            cmd.extend(["-profile:v", "high", "-level", "4.0"])
        # Audio handling
        if include_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.extend(["-an"])  # No audio
    elif output_ext == '.webm':
        # VP9 for WebM
        # VP9 quality scale is 0-63, where 0 is best. Map 100->0, 0->63
        vp9_quality = int(63 * (1 - quality / 100))
        vp9_quality = max(0, min(63, vp9_quality))
        cmd.extend(["-c:v", "libvpx-vp9", "-crf", str(vp9_quality), "-b:v", "0"])
        if convert_to_gray:
            cmd.extend(["-pix_fmt", "yuv420p"])
        # Audio handling
        if include_audio:
            cmd.extend(["-c:a", "libopus"])
        else:
            cmd.extend(["-an"])
    elif output_ext == '.mkv':
        # H.265 for MKV (better compression)
        crf_value = int(51 * (1 - quality / 100))
        crf_value = max(0, min(51, crf_value))
        cmd.extend(["-c:v", "libx265", "-preset", "slow", "-crf", str(crf_value)])
        if convert_to_gray:
            cmd.extend(["-pix_fmt", "yuv420p"])
        # Audio handling
        if include_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])
    elif output_ext == '.mov':
        # H.264 for MOV (better compatibility with QuickTime)
        crf_value = int(51 * (1 - quality / 100))
        crf_value = max(0, min(51, crf_value))
        cmd.extend(["-c:v", "libx264", "-preset", "slow", "-crf", str(crf_value)])
        if convert_to_gray:
            cmd.extend(["-pix_fmt", "yuv420p"])
        # Audio handling
        if include_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.extend(["-an"])
    elif output_ext == '.avi':
        # H.264 for AVI (widely compatible)
        crf_value = int(51 * (1 - quality / 100))
        crf_value = max(0, min(51, crf_value))
        cmd.extend(["-c:v", "libx264", "-preset", "slow", "-crf", str(crf_value)])
        if convert_to_gray:
            cmd.extend(["-pix_fmt", "yuv420p"])
        # Audio handling
        if include_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.extend(["-an"])
    else:
        # Default to H.265 for other formats (better compression)
        crf_value = int(51 * (1 - quality / 100))
        crf_value = max(0, min(51, crf_value))
        cmd.extend(["-c:v", "libx265", "-preset", "slow", "-crf", str(crf_value)])
        if convert_to_gray:
            cmd.extend(["-pix_fmt", "yuv420p"])
        # Audio handling
        if include_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])
    
    # Output file
    cmd.append(output_video)
    
    # Execute ffmpeg command with progress tracking
    try:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        last_reported_frame = -100
        frame_pattern = re.compile(r'frame=\s*(\d+)')
        stderr_lines = []
        
        # Read stderr line by line to track progress
        for line in process.stderr:
            stderr_lines.append(line)
            # Look for frame information in ffmpeg output
            match = frame_pattern.search(line)
            if match:
                current_frame = int(match.group(1))
                # Report progress every 100 frames
                if current_frame - last_reported_frame >= 100:
                    progress_pct = (current_frame / expected_output_frames * 100) if expected_output_frames > 0 else 0
                    print(f"   Processing frame {current_frame}/{expected_output_frames} ({progress_pct:.1f}%)")
                    last_reported_frame = current_frame
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            stderr_output = "".join(stderr_lines)
            error_msg = f"ffmpeg error for {input_video}: {stderr_output}"
            raise RuntimeError(error_msg)
        
        print(f"   Video processing complete. Output saved to: {output_video}")
        return output_video
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg error for {input_video}: {e.stderr if hasattr(e, 'stderr') else 'Unknown error'}"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Error processing video {input_video}: {str(e)}"
        raise RuntimeError(error_msg) from e


def create_small_video(video_path, video_filename=None, roi=None, downsample_factor=1, convert_to_gray=False, start_frame=0, end_frame=None, skip_frames=1, output_dir_path=None, quality=80, output_format=None, include_audio=False, enhance_contrast=False):
    """
    Create a processed video using ffmpeg with spatial cropping, downsampling, grayscale conversion,
    time trimming, and frame skipping.
    
    If video_filename is None, processes all video files in video_path directory.
    
    Args:
        video_path: Path to the directory containing the video file(s), or full path to video file
        video_filename: Name of the video file (if None, processes all videos in video_path)
        roi: Tuple (x, y, w, h) for spatial cropping, or None for no cropping
        downsample_factor: Factor to downsample by (e.g., 2 means half size). Must be >= 1
        convert_to_gray: If True, convert video to grayscale
        start_frame: Starting frame number (0-indexed)
        end_frame: Ending frame number (None for end of video)
        skip_frames: Keep every Nth frame (e.g., 2 means keep every 2nd frame). Must be >= 1
        output_dir_path: Output directory path (if None, saves in video_path)
        quality: Quality level from 0-100 (100 = no compression, 0 = maximum compression). Default: 80
        output_format: Output format extension (e.g., '.mp4', '.mov', '.mkv'). If None, uses same format as input
        include_audio: If False, exclude audio from output. Default: False
        enhance_contrast: If True, apply histogram equalization to expand contrast to full range. Default: False
        
    Returns:
        str or list: Path(s) to the output video file(s)
    """
    # Validate quality parameter
    if not (0 <= quality <= 100):
        raise ValueError(f"quality must be between 0 and 100, got {quality}")
    
    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}
    
    # Determine if processing single file or batch
    if video_filename is None:
        # Batch processing: get all video files in directory
        if not os.path.isdir(video_path):
            raise ValueError(f"video_path must be a directory when video_filename is None: {video_path}")
        
        video_files = [
            f for f in os.listdir(video_path)
            if os.path.isfile(os.path.join(video_path, f)) and 
            os.path.splitext(f.lower())[1] in video_extensions and
            not os.path.splitext(f)[0].endswith('_small')  # Exclude already processed _small files
        ]
        
        if not video_files:
            raise ValueError(f"No video files found in directory: {video_path}")
        
        # Determine output directory
        if output_dir_path is None:
            output_dir = video_path
        else:
            output_dir = output_dir_path
            os.makedirs(output_dir, exist_ok=True)
        
        # Process each video file
        output_paths = []
        for filename in video_files:
            input_video = os.path.join(video_path, filename)
            base_name, input_ext = os.path.splitext(filename)

            print(f"Processing {filename}")
            
            # Determine output extension
            if output_format is None:
                output_ext = input_ext
            else:
                output_ext = output_format if output_format.startswith('.') else f'.{output_format}'
            
            output_filename = f"{base_name}_small{output_ext}"
            output_video = os.path.join(output_dir, output_filename)
            
            try:
                result = _process_single_video(
                    input_video, output_video, roi, downsample_factor,
                    convert_to_gray, start_frame, end_frame, skip_frames, quality, include_audio, enhance_contrast
                )
                output_paths.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return output_paths
    else:
        # Single file processing
        input_video = os.path.join(video_path, video_filename)
        
        # Determine output directory and filename
        if output_dir_path is None:
            output_dir = video_path
        else:
            output_dir = output_dir_path
            os.makedirs(output_dir, exist_ok=True)
        
        base_name, input_ext = os.path.splitext(video_filename)
        
        # Determine output extension
        if output_format is None:
            output_ext = input_ext
        else:
            output_ext = output_format if output_format.startswith('.') else f'.{output_format}'
        
        output_filename = f"{base_name}_small{output_ext}"
        output_video = os.path.join(output_dir, output_filename)
        
        return _process_single_video(
            input_video, output_video, roi, downsample_factor,
            convert_to_gray, start_frame, end_frame, skip_frames, quality, include_audio, enhance_contrast
        )