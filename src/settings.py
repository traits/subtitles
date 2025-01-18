from dataclasses import dataclass, field
import subprocess
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    media_file: str
    root: Path = field(init=False)
    data_dir: Path = field(init=False)
    odir: Path = field(init=False)
    odir_frames: Path = field(init=False)
    odir_rois: Path = field(init=False)
    log_file: Path = field(init=False)
    log_frame_info: Path = field(init=False)
    ocr_result: Path = field(init=False)
    audio_result: Path = field(init=False)

    def __post_init__(self):
        self.root = self._get_git_root()
        self.data_dir = self.root / "data"
        self.media_file = self.data_dir / self.media_file

        self.odir = self.root / "_output" / self.media_file.stem
        self.odir_frames = self.odir / "frames"
        self.odir_rois = self.odir / "rois"

        self.log_file = self.odir / "ffmpeg.log"  # ffmpeg log file (loglevel 'debug') (Preprocessor)
        self.log_frame_info = self.odir / "frame_info.json"  # frame number and pts, parsed from log file (Preprocessor)
        self.ocr_result = self.odir / "ocr_result.json"  # result of vllm inference (OcrAnalyzer)
        self.audio_result = self.odir / "audio_result.json"  # result of llm inference (AudioAnalyzer)

        self.odir.mkdir(parents=True, exist_ok=True)
        self.odir_frames.mkdir(parents=True, exist_ok=True)
        self.odir_rois.mkdir(parents=True, exist_ok=True)

    def _get_git_root(self) -> Path:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        if not root:
            raise OSError(2, "file not found (no git root detected)")
        s = root.decode("utf-8").strip()
        return Path(s)
