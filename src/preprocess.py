import json
import re
import subprocess
from pathlib import Path

import cv2

from settings import Settings


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


def print_loop_state_modulo(i, loop_size, occ):
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
        self.settings = Settings()
        self.mkv_file = self.settings.mkv_file
        self.odir = self.settings.odir
        self.odir_frames = self.settings.odir_frames
        self.odir_rois = self.settings.odir_rois

    def run(self):
        self.extract_roi_images(10)

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

    def runFFMPEG(self, fps, log_file):
        # Extract video frames using ffmpeg
        command = [
            "ffmpeg",
            "-loglevel",
            "debug",
            "-i",
            self.mkv_file.as_posix(),
            "-vf",
            f"fps={fps}",
            "-fps_mode",
            "auto",
            "-frame_pts",
            "1",
            str(self.odir_frames / "%05d.png"),
        ]

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        ffmpeg_log = result.stderr
        with open(log_file, "w") as f:
            f.write(ffmpeg_log)

    def convertFFMPEGLog(self, log_file) -> list:
        # Regular expression to match the desired log entries
        log_pattern = re.compile(r"\[Parsed_fps_\d+ @ [0-9a-fA-F]+] (Read frame .*|Dropping frame .*|Writing frame .*)")

        # Regular expression to extract PTS values
        pts_pattern = re.compile(r"in pts (\d+), out pts (\d+)|with pts (\d+)")

        # List to store the extracted log entries
        result = []

        # Counter for 'read' frames
        read_frame_counter = 0

        # Read the log file and extract matching lines
        with open(log_file, "r") as f:
            for line in f:
                match = log_pattern.match(line)
                if match:
                    entry = match.group(1)
                    # Determine the type
                    if entry.startswith("Read frame"):
                        entry_type = "read"
                    elif entry.startswith("Dropping frame"):
                        entry_type = "drop"
                    elif entry.startswith("Writing frame"):
                        entry_type = "write"
                    else:
                        continue  # Skip if the type is not recognized

                    # Extract PTS values
                    pts_match = pts_pattern.search(entry)
                    if pts_match:
                        if pts_match.group(1) and pts_match.group(2):
                            in_pts = int(pts_match.group(1))
                            out_pts = int(pts_match.group(2))
                        elif pts_match.group(3):
                            in_pts = int(pts_match.group(3))
                            out_pts = None  # No out PTS in this case
                        else:
                            continue  # Skip if no PTS values are found

                        # Create a dictionary entry with the updated keys and frame index
                        entry_dict = {"type": entry_type, "in_pts": in_pts, "out_pts": out_pts}

                        # Add 'frame' only for 'read' entries
                        if entry_type == "read":
                            entry_dict["frame"] = read_frame_counter
                            read_frame_counter += 1  # Increment the frame counter

                        result.append(entry_dict)
        return result

    def filterFrameInfo(self, data, output_file):
        result = []
        last_i = len(data) - 1
        i = 0
        while i <= last_i:
            if i < last_i:
                v = data[i]
                n = data[i + 1]
                if v["type"] == "read" and n["type"] == "drop":
                    if v["out_pts"] == n["in_pts"]:
                        i += 2
                        continue
                else:
                    result.append(v)
            i += 1

        # result = [{"frame": v["frame"], "in_pts": v["in_pts"], "out_pts": v["out_pts"]} for v in result if v["type"] == "read"]
        # better: this line uses the fact, that out_pts == list index for the new list
        result = [{"frame": v["frame"], "in_pts": v["in_pts"]} for v in result if v["type"] == "read"]
        # Save the extracted entries as a JSON list
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    def extract_roi_images(self, fps=1):
        log_file = self.odir / "ffmpeg.log"
        self.runFFMPEG(fps, log_file)
        frames = self.convertFFMPEGLog(log_file)
        fi_file = self.odir / "frame_info.json"
        self.filterFrameInfo(frames, fi_file)

        # Calculate the number of .png files in the output directory
        frames = sorted(list(self.odir_frames.glob("*.png")))
        num_frames = len(frames)
        print(f"Number of frames to process: {num_frames}")

        # Process each frame to detect and extract subtitles
        frame_idx = 0
        frame_size = self.get_video_dimensions()
        print(f"{frame_size=}")
        roi_x, roi_y = self.get_roi(frame_size)
        print(f"{roi_x=} {roi_y=}")

        for i, frame_path in enumerate(frames):
            print_loop_state_modulo(i, num_frames, 100)

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
            subtitle_path = self.odir_rois / f"{frame_idx:05d}.png"
            cv2.imwrite(str(subtitle_path), subtitle_gray)

            frame_idx += 1
