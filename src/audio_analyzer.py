from pathlib import Path

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
            return_timestamps=True,
            chunk_length_s=30,  # Force 30-second chunks
        )

    def run(self):
        """Run audio analysis on the media file from settings."""
        if not self.settings.media_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.settings.media_file}")

        result = self.pipe(
            str(self.settings.media_file),
            generate_kwargs={
                "language": "zh",  # Source language (Chinese)
                "task": "translate",  # Translate to English
                "forced_decoder_ids": None  # Move here to avoid conflict
            }
        )

        # Create structured JSON results
        json_results = []
        
        if isinstance(result["text"], str):
            # Short audio case - single segment
            json_results.append({
                "english": result["text"],  # Translated text
                "pts": [0.0, 0.0]  # Dummy timestamps
            })
        else:
            # Long audio case - multiple segments
            for chunk in result["text"]:
                json_results.append({
                    "english": chunk["text"],  # Translated text
                    "pts": [float(chunk['timestamp'][0]), float(chunk['timestamp'][1])]
                })
        
        # Save results as JSON to output directory
        import json
        with open(self.settings.audio_result, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
            
        return json_results
