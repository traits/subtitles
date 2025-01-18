from abc import ABC, abstractmethod
from pathlib import Path
import json

from settings import Settings


class BaseAnalyzer(ABC):
    """Abstract base class for analyzers."""
    
    def __init__(self, result_file: Path):
        """
        Initialize the analyzer with common settings.
        
        Args:
            result_file: Path to store analysis results
        """
        self.result_file = result_file
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> dict:
        """Load prompts from JSON file."""
        prompts_path = Settings.data_dir / "prompts.json"
        with open(prompts_path, "r") as f:
            return json.load(f)
            
    @abstractmethod
    def run(self):
        """Run the analysis. Must be implemented by subclasses."""
        pass
