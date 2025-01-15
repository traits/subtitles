import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class AudioAnalyzer:
    def __init__(self, audio_file: Path):
        """Initialize the analyzer with a FLAC audio file."""
        self.audio_file = audio_file
        self.audio_data = None
        self.sample_rate = None
        
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """Load the FLAC audio file using librosa."""
        self.audio_data, self.sample_rate = librosa.load(
            self.audio_file, 
            sr=None,  # Use native sample rate
            mono=True  # Convert to mono if needed
        )
        return self.audio_data, self.sample_rate
        
    def get_duration(self) -> float:
        """Get the duration of the audio in seconds."""
        if self.audio_data is None:
            self.load_audio()
        return librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
        
    def get_rms_energy(self) -> np.ndarray:
        """Calculate the RMS energy of the audio signal."""
        if self.audio_data is None:
            self.load_audio()
        return librosa.feature.rms(y=self.audio_data)
        
    def detect_silence(self, threshold_db: float = -40.0) -> np.ndarray:
        """Detect silent portions of the audio.
        
        Args:
            threshold_db: Silence threshold in decibels
            
        Returns:
            Array of time intervals where audio is below threshold
        """
        if self.audio_data is None:
            self.load_audio()
            
        # Convert to dB
        rms = self.get_rms_energy()
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find silent intervals
        silent_intervals = np.where(rms_db < threshold_db)[0]
        return silent_intervals
        
    def analyze(self) -> dict:
        """Run basic audio analysis and return results as a dictionary."""
        if self.audio_data is None:
            self.load_audio()
            
        return {
            'duration': self.get_duration(),
            'sample_rate': self.sample_rate,
            'rms_energy': self.get_rms_energy(),
            'silent_intervals': self.detect_silence()
        }
