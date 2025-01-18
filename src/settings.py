import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    media_file: str
    root: Path = field(init=False)
    data_dir: Path = field(init=False)
    out_dir: Path = field(init=False)
    out_frames: Path = field(init=False)
    out_rois: Path = field(init=False)
    log_file: Path = field(init=False)
    log_frame_info: Path = field(init=False)
    result_ocr: Path = field(init=False)
    result_audio: Path = field(init=False)

    def __post_init__(self):
        self.root = self._get_git_root()
        self.data_dir = self.root / "data"
        self.media_file = self.data_dir / self.media_file

        self.out_dir = self.root / "_output" / self.media_file.stem
        self.out_frames = self.out_dir / "frames"
        self.out_rois = self.out_dir / "rois"

        self.log_file = self.out_dir / "ffmpeg.log"  # ffmpeg log file (loglevel 'debug') (Preprocessor)
        self.log_frame_info = self.out_dir / "frame_info.json"  # frame number and pts, parsed from log file (Preprocessor)
        self.result_ocr = self.out_dir / "result_ocr.json"  # result of vllm inference (OcrAnalyzer)
        self.result_audio = self.out_dir / "result_audio.json"  # result of llm inference (AudioAnalyzer)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_frames.mkdir(parents=True, exist_ok=True)
        self.out_rois.mkdir(parents=True, exist_ok=True)

    def _get_git_root(self) -> Path:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        if not root:
            raise OSError(2, "file not found (no git root detected)")
        s = root.decode("utf-8").strip()
        return Path(s)
