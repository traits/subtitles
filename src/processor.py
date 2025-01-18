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
        settings.initialize_dirs()
        
        # Run both OCR and audio processing
        ocr_analyzer = OcrAnalyzer()
        ocr_analyzer.run()
        
        audio_analyzer = AudioAnalyzer(model_id="openai/whisper-large-v3")
        audio_analyzer.run()
        
        # Process combined subtitles
        postprocessor = PostProcessor()
        postprocessor.run(ProcessType.AUDIO)  # This will now generate all subtitle files
