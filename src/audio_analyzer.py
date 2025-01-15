import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path

class AudioAnalyzer:
    def __init__(self, settings, model_id="openai/whisper-large-v3"):
        """Initialize the audio analyzer with Whisper model.
        
        Args:
            settings: Settings object containing media file path
            model_id: Model identifier for Whisper
        """
        self.settings = settings
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
            return_timestamps=True
        )

    def run(self):
        """Run audio analysis on the media file from settings."""
        if not self.settings.media_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.settings.media_file}")
            
        result = self.pipe(str(self.settings.media_file))
        
        # Save results to output directory
        with open(self.settings.audio_result, "w") as f:
            if isinstance(result["text"], str):
                # Short audio case
                f.write(result["text"])
                return result["text"]
            else:
                # Long audio case with timestamps
                text_with_timestamps = "\n".join(
                    f"[{chunk['timestamp'][0]:.2f}-{chunk['timestamp'][1]:.2f}] {chunk['text']}"
                    for chunk in result["text"]
                )
                f.write(text_with_timestamps)
                return text_with_timestamps
