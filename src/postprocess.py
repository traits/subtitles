import json
from enum import Enum
from pathlib import Path

from settings import Settings


class ProcessType(Enum):
    """Type of processing being performed"""

    OCR = 1
    AUDIO = 2


class PostProcessor:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.media_file = self.settings.media_file
        self.odir = self.settings.odir
        self.ocr_result = self.settings.ocr_result
        self.audio_result = self.settings.audio_result
        self.frameinfo_file = self.settings.log_frame_info
        self.sub_file_ocr = self.odir / f"{self.media_file.stem}.sub"

    def run(self):
        self.writeSubFile()

    def mergeSubTitleInfo(self) -> list:
        with open(self.ocr_result, "r", encoding="utf8") as f:
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
        info = self.mergeSubTitleInfo()

        with open(self.sub_file_ocr, "w", encoding="utf8") as f:
            last_i = len(info) - 1
            last_chinese = ""
            last_english = ""
            for i, v in enumerate(info):
                if i < last_i:
                    if text := v.get("english"):
                        if (ctext := v.get("chinese")) == last_chinese:
                            f.write(f"{{{v['frame']}}}{{{info[i+1]['frame']}}}{last_english}\n")
                        else:
                            f.write(f"{{{v['frame']}}}{{{info[i+1]['frame']}}}{text}\n")
                            last_english = text
                            last_chinese = ctext
