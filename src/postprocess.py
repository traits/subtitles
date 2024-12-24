import json
from pathlib import Path

from settings import Settings


class PostProcessor:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.video_file = self.settings.video_file
        self.odir = self.settings.odir
        self.ocr_result = self.settings.ocr_result
        self.frameinfo_file = self.settings.log_frame_info
        self.info_file = self.odir / "info.json"
        self.sub_file = self.odir / f"{self.video_file.stem}.sub"

    def run(self):
        result = self.mergeSubTitleInfo()
        with open(self.info_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
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
        with open(self.info_file, "r", encoding="utf8") as f:
            info = json.load(f)
        with open(self.sub_file, "w", encoding="utf8") as f:
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
