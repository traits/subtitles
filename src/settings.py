import subprocess
from pathlib import Path


class Settings:
    def __init__(self):

        self.root = self._get_git_root()

        self.mkv_file = self.root / "data" / "AiO.mkv"  # Replace with your MKV file path

        self.odir = self.root / "_output"
        self.odir_frames = self.odir / "frames"
        self.odir_rois = self.odir / "rois"

        self.odir.mkdir(parents=True, exist_ok=True)
        self.odir_frames.mkdir(parents=True, exist_ok=True)
        self.odir_rois.mkdir(parents=True, exist_ok=True)

    def _get_git_root(self) -> Path:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        if not root:
            raise OSError(2, "file not found (no git root detected)")
        s = root.decode("utf-8").strip()
        return Path(s)
