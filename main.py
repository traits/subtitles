import json
import subprocess
from pathlib import Path

import cv2


def get_video_dimensions(mkv_file):
    width = height = -1
    vcap = cv2.VideoCapture(mkv_file)
    if vcap.isOpened():
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        vcap.release()
    return int(width), int(height)


def get_roi(vdim: tuple):
    y_offset = 0.02  # relative to frame height from bottom
    y_size = 0.2  # height of roi as fraction of frame height
    x_size = 0.7  # The roi's x extension as fraction of frame width
    w = vdim[0]
    h = vdim[1]
    x_begin = int((1 - x_size) * w / 2.0)
    x_end = int(w - x_begin)
    y_begin = int(y_offset * h)
    y_end = int((y_offset + y_size) * h)

    return (x_begin, x_end), (y_begin, y_end)


def extract_subtitles(mkv_file):
    # Create the output directory if it doesn't exist
    odir = Path("_subtitles")
    odir.mkdir(parents=True, exist_ok=True)
    odir_frames = odir / "frames"
    odir_frames.mkdir(parents=True, exist_ok=True)
    odir_rois = odir / "rois"
    odir_rois.mkdir(parents=True, exist_ok=True)

    # Extract video frames using ffmpeg
    subprocess.run(["ffmpeg", "-i", mkv_file, "-vf", "fps=1", str(odir_frames / "%05d.png")], check=True)

    # Calculate the number of .png files in the output directory
    frames = sorted(list(odir_frames.glob("*.png")))
    num_frames = len(frames)
    print(f"Number of frames to process: {num_frames}")

    # Process each frame to detect and extract subtitles
    subtitles = []
    frame_count = 1
    frame_size = get_video_dimensions(mkv_file)
    print(f"{frame_size=}")
    roi_x, roi_y = get_roi(frame_size)
    print(f"{roi_x=} {roi_y=}")

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Video dimensions
        height, width = gray.shape

        # y range in opencv coordinates:
        roi_y_cv = (height - roi_y[1], height - roi_y[0])

        # Crop the subtitle region
        subtitle_gray = gray[roi_y_cv[0] : roi_y_cv[1], roi_x[0] : roi_x[1]]

        # Save the cropped subtitle image
        subtitle_path = odir_rois / f"{frame_count:05d}.png"
        cv2.imwrite(str(subtitle_path), subtitle_gray)

        # Store the timestamp and associated image path
        subtitles.append({"timestamp": f"{frame_count:05d}", "image": subtitle_path.name})

        frame_count += 1

    # Write the timestamps and associated image paths to a JSON file
    with open(odir / "subtitles.json", "w") as json_file:
        json.dump(subtitles, json_file, indent=4)


if __name__ == "__main__":
    mkv_file = "data/AiO.mkv"  # Replace with your MKV file path
    extract_subtitles(mkv_file)
