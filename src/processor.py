from pathlib import Path

from audio_analyzer import AudioAnalyzer
from ocr_analyzer import OcrAnalyzer
from postprocess import PostProcessor, ProcessType
from preprocess import VideoPreprocessor
from settings import Settings


class Processor:
    def __init__(self):
        Settings.initialize_dirs()

    def run(self):
        # Run both OCR and audio processing

        # ocr_analyzer = OcrAnalyzer()
        # ocr_analyzer.run()

        audio_analyzer = AudioAnalyzer()
        audio_analyzer.run()

        # # Process combined subtitles
        # postprocessor = PostProcessor()
        # postprocessor.run(ProcessType.BOTH)  # Changed from AUDIO to BOTH
