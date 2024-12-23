import json
import re
import subprocess
from pathlib import Path

import cv2

from settings import Settings


class PostProcessor:
    def __init__(self):
        self.settings = Settings()
        self.mkv_file = self.settings.mkv_file
        self.odir = self.settings.odir
        self.subtitles_file = self.odir / "subtitles.json"
        self.frameinfo_file = self.odir / "frame_info.json"
        self.info_file = self.odir / "info.json"
        self.sub_file = self.odir / self.mkv_file.with_suffix(".sub")

    def run(self):
        result = self.mergeSubTitleInfo()
        with open(self.info_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.writeSubFile()

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

    def mergeSubTitleInfo(self) -> list:
        with open(self.subtitles_file, "r", encoding="utf8") as f:
            sinfo = json.load(f)
        with open(self.frameinfo_file, "r") as f:
            finfo = json.load(f)

        ls = len(sinfo)
        lf = len(finfo)
        if ls != lf:
            print(f"{ls=} {lf=}")
            return []

        result = []
        for si, fi in zip(sinfo, finfo):
            elem = si
            if elem == None:
                elem = {}

            elem["pts"] = fi["in_pts"]
            elem["frame"] = fi["frame"]
            result.append(elem)

        return result

    def writeSubFile(self):
        with open(self.info_file, "r", encoding="utf8") as f:
            info = json.load(f)
        with open(self.sub_file, "w", encoding="utf8") as f:
            last_i = len(info) - 1
            for i, v in enumerate(info):
                if i < last_i:
                    if text := v.get("english"):
                        f.write(f"{{{v['frame']}}}{{{info[i+1]['frame']}}}{text}\n")
