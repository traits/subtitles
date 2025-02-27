import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class Settings:
    @staticmethod
    def _get_git_root() -> Path:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        if not root:
            raise OSError(2, "file not found (no git root detected)")
        s = root.decode("utf-8").strip()
        return Path(s)

    media_base_name: str = "AiO-ep19"  # Default media file base name
    # Initialize paths once when class is loaded
    root: Path = _get_git_root()
    data_dir: Path = root / "data"
    video_file: Path = data_dir / f"{media_base_name}.mkv"
    audio_file: Path = data_dir / f"{media_base_name}.flac"
    prompts_path = data_dir / "prompts.json"

    out_dir: Path = root / "_output" / media_base_name
    out_frames: Path = out_dir / "frames"
    out_rois: Path = out_dir / "rois"

    out_diarization: Path = out_dir / "diarization.txt"

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
