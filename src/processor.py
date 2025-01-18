from pathlib import Path

from audio_analyzer import AudioAnalyzer
from ocr_analyzer import OcrAnalyzer
from postprocess import PostProcessor, ProcessType
from preprocess import VideoPreprocessor
from settings import settings


class Processor:

    def __init__(self):
        settings.initialize_dirs()

    def run(self):
        # preprocess = PreProcessor(self._settings)
        # preprocess.run()
        # analyzer = OcrAnalyzer(self._settings)
        # analyzer.run()
        # analyzer = AudioAnalyzer(self._settings, model_id="openai/whisper-large-v3")
        # analyzer.run()
        postprocessor = PostProcessor(self._settings)
        postprocessor.run(ProcessType.AUDIO)
