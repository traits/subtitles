import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path

class AudioAnalyzer:
    def __init__(self, model_id="openai/whisper-large-v3"):
        """Initialize the audio analyzer with Whisper model."""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe_audio(self, flac_path: Path) -> str:
        """Transcribe audio from a FLAC file.
        
        Args:
            flac_path: Path to the FLAC audio file
            
        Returns:
            Transcribed text from the audio
        """
        if not flac_path.exists():
            raise FileNotFoundError(f"Audio file not found: {flac_path}")
            
        result = self.pipe(str(flac_path))
        return result["text"]

if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    audio_file = Path("audio_file.flac")  # Replace with your FLAC file path
    transcription = analyzer.transcribe_audio(audio_file)
    print(transcription)
