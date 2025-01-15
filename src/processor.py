from pathlib import Path

from audio_analyzer import AudioAnalyzer
from ocr_analyzer import OcrAnalyzer
from postprocess import PostProcessor
from preprocess import VideoPreprocessor
from settings import Settings


class Processor:
    def __init__(self):
        # self._settings = Settings("AiO-ep19.mkv")  # expected in "<GITROOT>/data" dir
        self._settings = Settings("AiO-ep19.flac")

    def run(self):
        # preprocess = PreProcessor(self._settings)
        # preprocess.run()
        # analyzer = OcrAnalyzer(self._settings)
        # analyzer.run()
        analyzer = AudioAnalyzer(self._settings, model_id="openai/whisper-large-v3")
        analyzer.run()
        # postprocessor = PostProcessor(self._settings)
        # postprocessor.run()
