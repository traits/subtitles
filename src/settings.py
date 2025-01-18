import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class Settings:
    media_file: str = "AiO-ep19.mkv"  # Default media file name
    
    # Initialize paths once when class is loaded
    root: Path = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip())
    data_dir: Path = root / "data"
    media_path: Path = data_dir / media_file
    
    out_dir: Path = root / "_output" / media_path.stem
    out_frames: Path = out_dir / "frames"
    out_rois: Path = out_dir / "rois"
    
    log_file: Path = out_dir / "ffmpeg.log"  # ffmpeg log file (loglevel 'debug') (Preprocessor)
    log_frame_info: Path = out_dir / "frame_info.json"  # frame number and pts, parsed from log file (Preprocessor)
    result_ocr: Path = out_dir / "result_ocr.json"  # result of vllm inference (OcrAnalyzer)
    result_audio: Path = out_dir / "result_audio.json"  # result of llm inference (AudioAnalyzer)

    @classmethod
    def initialize_dirs(cls):
        """Create output directories if they don't exist"""
        cls.out_dir.mkdir(parents=True, exist_ok=True)
        cls.out_frames.mkdir(parents=True, exist_ok=True)
        cls.out_rois.mkdir(parents=True, exist_ok=True)

    def _get_git_root(self) -> Path:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        if not root:
            raise OSError(2, "file not found (no git root detected)")
        s = root.decode("utf-8").strip()
        return Path(s)
