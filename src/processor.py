from pathlib import Path

from analyzer import Analyzer
from postprocess import PostProcessor
from preprocess import PreProcessor
from settings import Settings


class Processor:
    def __init__(self):
        self._settings = Settings("AiO-ep19.mkv")  # expected in "<GITROOT>/data" dir

    def run(self):
        # preprocess = PreProcessor(self._settings)
        # preprocess.run()
        # analyzer = Analyzer(self._settings)
        # analyzer.run()
        postprocessor = PostProcessor(self._settings)
        postprocessor.run()
