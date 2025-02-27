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
        # ocr_analyzer = OcrAnalyzer(model_id="Qwen/Qwen2-VL-7B-Instruct")
        # ocr_analyzer.run()

        audio_analyzer = AudioAnalyzer(model_id="openai/whisper-large-v3")
        audio_analyzer.run()

        # Process combined subtitles
        postprocessor = PostProcessor()
        postprocessor.run(ProcessType.BOTH)  # Changed from AUDIO to BOTH
