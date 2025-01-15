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
        self.sub_file_ocr = self.odir / f"{self.media_file.stem}_ocr.sub"

    def run(self, process_type: ProcessType = ProcessType.OCR):
        """Run subtitle file generation for the specified processing type.
        
        Args:
            process_type: Type of processing to generate subtitles for (OCR or AUDIO)
        """
        if process_type == ProcessType.OCR:
            self.writeOcrSubFile()
        elif process_type == ProcessType.AUDIO:
            self.writeAudioSubFile()
        else:
            raise ValueError(f"Unknown process type: {process_type}")

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

    def writeOcrSubFile(self):
        """Create subtitle file from OCR analysis results."""
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

    def writeAudioSubFile(self):
        """Create subtitle file from audio analysis results in SRT format."""
        sub_file_audio = self.odir / f"{self.media_file.stem}_audio.srt"

        with open(self.audio_result, "r", encoding="utf8") as f:
            audio_info = json.load(f)

        def ms_to_srt_time(ms):
            """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
            hours = ms // 3_600_000
            ms %= 3_600_000
            minutes = ms // 60_000
            ms %= 60_000
            seconds = ms // 1_000
            ms %= 1_000
            return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

        with open(sub_file_audio, "w", encoding="utf8") as f:
            last_i = len(audio_info) - 1
            last_english = ""
            subtitle_index = 1
            
            for i, v in enumerate(audio_info):
                if i < last_i:
                    if text := v.get("english"):
                        if text != last_english:
                            # Write SRT entry
                            start_time = ms_to_srt_time(v['pts'])
                            end_time = ms_to_srt_time(audio_info[i+1]['pts'])
                            
                            f.write(f"{subtitle_index}\n")
                            f.write(f"{start_time} --> {end_time}\n")
                            f.write(f"{text}\n\n")
                            
                            subtitle_index += 1
                            last_english = text
