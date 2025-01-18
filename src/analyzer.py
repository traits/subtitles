import json
from abc import ABC, abstractmethod
from pathlib import Path

from settings import Settings


class BaseAnalyzer(ABC):
    """Abstract base class for analyzers."""

    def __init__(self):
        """
        Initialize the analyzer with common settings.
        """
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load prompts from JSON file."""
        with open(Settings.prompts_path, "r") as f:
            return json.load(f)

    @abstractmethod
    def run(self):
        """Run the analysis. Must be implemented by subclasses."""
        pass
