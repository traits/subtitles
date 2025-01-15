import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
        
        # Initialize model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline with timestamp settings
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def run(self):
        """Run audio analysis on the media file."""
        result = self.pipe(str(self.settings.media_file))
        return result
