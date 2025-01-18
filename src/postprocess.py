import json
from enum import Enum
from pathlib import Path

from settings import Settings


class ProcessType(Enum):
    """Type of processing being performed"""
    NONE = 0
    OCR = 1
    AUDIO = 2
    BOTH = OCR | AUDIO  # Combined processing type

    def __or__(self, other):
        return ProcessType(self.value | other.value)

    def __or__(self, other):
        return ProcessType(self.value | other.value)


class PostProcessor:

    def __init__(self):
        # Add stream-specific output files
        self.sub_files = {
            "ocr": Settings.out_dir / f"{Settings.media_path.stem}_ocr.sub",
            "audio": Settings.out_dir / f"{Settings.media_path.stem}_audio.srt",
            "combined": Settings.out_dir / f"{Settings.media_path.stem}.srt",
        }

    def run(self, process_type: ProcessType = ProcessType.OCR):
        """Run subtitle file generation for the specified processing type.
        
        Args:
            process_type: Type of processing to generate subtitles for (OCR, AUDIO, or BOTH)
        """
        if process_type == ProcessType.NONE:
            return
            
        if process_type & ProcessType.OCR:
            self.writeOcrSubFile()
            
        if process_type & ProcessType.AUDIO:
            self.writeAudioSubFile()
            self.writeCombinedSubFile()

    def mergeSubTitleInfo(self) -> list:
        with open(Settings.result_ocr, "r", encoding="utf8") as f:
            sinfo = json.load(f)
        with open(Settings.log_frame_info, "r") as f:
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
        """Create subtitle file from OCR analysis results in SUB format (frame based)."""
        info = self.mergeSubTitleInfo()

        with open(self.sub_files["ocr"], "w", encoding="utf8") as f:
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

    def ms_to_srt_time(self, ms):
        """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
        hours = ms // 3_600_000
        ms %= 3_600_000
        minutes = ms // 60_000
        ms %= 60_000
        seconds = ms // 1_000
        ms %= 1_000
        return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

    def writeCombinedSubFile(self):
        """Create a combined subtitle file with both OCR and audio streams."""
        
        def write_subtitle_stream(f, stream_type: str, subtitle_data: list, color_code: str):
            """Nested function to write a subtitle stream.
            
            Args:
                f: File handle to write to
                stream_type: Type of stream (OCR or AUDIO)
                subtitle_data: List of subtitle entries
                color_code: HTML color code for the stream
            """
            nonlocal subtitle_index
            last_i = len(subtitle_data) - 1
            for i, v in enumerate(subtitle_data):
                if i < last_i:
                    if text := v.get("english"):
                        start_time = self.ms_to_srt_time(v['pts'])
                        end_time = self.ms_to_srt_time(subtitle_data[i+1]['pts'])
                        
                        f.write(f"{subtitle_index}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"<font color=\"{color_code}\">[{stream_type}] {text}</font>\n\n")
                        subtitle_index += 1

        # Load OCR data
        ocr_info = self.mergeSubTitleInfo()
        # Load audio data
        with open(Settings.result_audio, "r", encoding="utf8") as f:
            audio_info = json.load(f)

        with open(self.sub_files["combined"], "w", encoding="utf8") as f:
            subtitle_index = 1
            
            # Write OCR stream
            write_subtitle_stream(f, "OCR", ocr_info, "#00FF00")
            
            # Write Audio stream
            write_subtitle_stream(f, "Audio", audio_info, "#FFFF00")

    def writeAudioSubFile(self):
        """Create subtitle file from audio analysis results in SRT format (timestamp based)."""
        with open(Settings.result_audio, "r", encoding="utf8") as f:
            audio_info = json.load(f)

        with open(self.sub_files["audio"], "w", encoding="utf8") as f:
            subtitle_index = 1
            # Filter out duplicate consecutive entries
            filtered_audio = []
            last_english = ""
            for entry in audio_info:
                if entry.get("english") != last_english:
                    filtered_audio.append(entry)
                    last_english = entry.get("english")
            
            # Use the helper method to write the stream
            self._write_subtitle_stream(f, "Audio", filtered_audio, "#FFFFFF", subtitle_index)
