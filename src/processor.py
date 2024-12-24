from pathlib import Path

from analyzer import Analyzer
from postprocess import PostProcessor
from preprocess import PreProcessor
from settings import Settings


class Processor:
    def __init__(self):
        self._settings = Settings()

    def run(self):
        # preprocess = PreProcessor(self._settings)
        # preprocess.run()
        # analyzer = Analyzer(self._settings)
        # analyzer.run()
        postprocessor = PostProcessor(self._settings)
        postprocessor.run()
