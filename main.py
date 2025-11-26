# %% [markdown]
""" Select region-of-interest from a video file """
import video_transforms as vt
import os

# Path to the directory containing the video file(s)
video_dir_path = os.path.join("/", "Users", "mmchenry", "Documents", "Video", "practice edits")

# Name of the video file to process
video_filename = "CF_neo_3_5dot5_C001H001S0005.mp4"

# Find region of interest 
roi = vt.select_roi(video_dir_path, video_filename)


# %% [markdown]
""" Create a small video from the original video """
import video_transforms as vt
import os

# Path to the directory containing the video file(s)
video_dir_path = os.path.join("/", "Users", "mmchenry", "Documents", "Video", "practice edits")
# Name of the video file, if you want to process single video
video_filename = None
# Region of interest to crop the video (None if you don't want to crop)
roi = (463, 379, 493, 183)
# int, default=1. How much to downsample video (2 = half size).
downsample_factor = 2
# bool, default=False. Convert video to grayscale if True.
convert_to_gray = False
# int, default=0. First frame to include (0-indexed).
start_frame = 0
# int or None, default=None. Last frame + 1 to include, or None for all frames.
end_frame = None
# int, default=1. Keep every Nth frame (1 = keep all frames).
skip_frames = 1
# str or None, default=None. Output directory path; None = same as input folder.
output_dir_path = None
# int, default=80. 0 (max compression) to 100 (no compression).
quality = 80
# str or None, default=None. E.g. 'mp4', 'avi', etc. None: same as input.
output_format = None
# bool, default=False. Include audio in output video.
include_audio = False
# bool, default=False. Apply histogram equalization for contrast if True.
enhance_contrast = False

# Create a small video from the original video
small_video_path = vt.create_small_video(video_dir_path, 
    video_filename=video_filename, roi=roi, downsample_factor=downsample_factor, convert_to_gray=convert_to_gray, start_frame=start_frame, end_frame=end_frame, skip_frames=skip_frames, output_dir_path=output_dir_path, quality=quality, output_format=output_format, include_audio=include_audio, enhance_contrast=enhance_contrast)

# %%
