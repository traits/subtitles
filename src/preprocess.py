import json
import subprocess
from pathlib import Path

import cv2


def print_loop_state(i, loop_size, occ):
    """
    Prints the current loop state as (i/loop_size) at evenly distributed occurrences.

    :param i: Current index of the loop.
    :param loop_size: Total size of the loop.
    :param occ: Number of times to print the loop state, evenly distributed.
    """
    if loop_size == 0:
        return  # Nothing to do if the loop size is zero

    # Calculate the step size to evenly distribute the print statements
    step = max(1, loop_size // occ)

    # Print the loop state only at the calculated intervals
    if i % step == 0:
        print(f"({i}/{loop_size})")


def print_loop_state_with_modulo(i, loop_size, occ):
    """
    Prints the current loop state as (i/loop_size) whenever i modulo occ is zero,
    and always prints the last iteration.

    :param i: Current index of the loop.
    :param loop_size: Total size of the loop.
    :param occ: An integer specifying the interval at which to print the loop state.
    """
    if loop_size == 0:
        return  # Nothing to do if the loop size is zero

    # Check if the current index matches the modulo condition or if it's the last iteration
    if i % occ == 0 or i == loop_size - 1:
        print(f"({i}/{loop_size})")


class PreProcessor:
    def __init__(self):

        root = self.get_git_root()

        self.mkv_file = root / "data" / "AiO.mkv"  # Replace with your MKV file path

        self.odir = root / "_output"
        self.odir_frames = self.odir / "frames"
        self.odir_rois = self.odir / "rois"

        self.odir.mkdir(parents=True, exist_ok=True)
        self.odir_frames.mkdir(parents=True, exist_ok=True)
        self.odir_rois.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.extract_roi_images(10)

    def get_git_root(self) -> Path:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        if not root:
            raise OSError(2, "file not found (no git root detected)")
        s = root.decode("utf-8").strip()
        return Path(s)

    def get_video_dimensions(self):
        width = height = -1
        vcap = cv2.VideoCapture(self.mkv_file.as_posix())
        if vcap.isOpened():
            width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            vcap.release()
        return int(width), int(height)

    def get_roi(self, vdim: tuple):
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

    def extract_roi_images(self, fps=1):
        # Extract video frames using ffmpeg
        subprocess.run(
            ["ffmpeg", "-i", self.mkv_file.as_posix(), "-vf", f"fps={fps}", str(self.odir_frames / "%05d.png")], check=True
        )

        # Calculate the number of .png files in the output directory
        frames = sorted(list(self.odir_frames.glob("*.png")))
        num_frames = len(frames)
        print(f"Number of frames to process: {num_frames}")

        # Process each frame to detect and extract subtitles
        subtitles = []
        frame_count = 1
        frame_size = self.get_video_dimensions()
        print(f"{frame_size=}")
        roi_x, roi_y = self.get_roi(frame_size)
        print(f"{roi_x=} {roi_y=}")

        for i, frame_path in enumerate(frames):
            print_loop_state(i, num_frames, 100)
            # print_loop_state_with_modulo(i, num_frames, 100)

            img = cv2.imread(str(frame_path))
            gray = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Video dimensions
            # height, width = gray.shape
            height = frame_size[1]

            # y range in opencv coordinates:
            roi_y_cv = (height - roi_y[1], height - roi_y[0])

            # Crop the subtitle region
            subtitle_gray = gray[roi_y_cv[0] : roi_y_cv[1], roi_x[0] : roi_x[1]]

            # Save the cropped subtitle image
            subtitle_path = self.odir_rois / f"{frame_count:05d}.png"
            cv2.imwrite(str(subtitle_path), subtitle_gray)

            # Store the timestamp and associated image path
            subtitles.append({"timestamp": f"{frame_count:05d}", "image": subtitle_path.name})

            frame_count += 1

        # Write the timestamps and associated image paths to a JSON file
        with open(self.odir / "subtitles.json", "w") as json_file:
            json.dump(subtitles, json_file, indent=4)
