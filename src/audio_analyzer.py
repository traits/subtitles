import json
from pathlib import Path

import librosa
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from analyzer import BaseAnalyzer
from settings import Settings


class Translator:
    """Specialized translator for converting Chinese text to English"""
    
    def __init__(self, model_id="Qwen/Qwen2-7B-Instruct"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side='left'  # Required for proper generation with decoder-only models
        )
        
    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of Chinese texts to English"""
        # Add translation instruction prompt
        prompts = [
            f"Translate this Chinese text to English: {text}" 
            for text in texts
        ]
        
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class AudioAnalyzer(BaseAnalyzer):
    def __init__(self, model_id="openai/whisper-large-v3"):
        """Initialize the audio analyzer with Whisper model.
        
        Args:
            model_id: Model identifier for Whisper
        """
        super().__init__()

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
            chunk_length_s=30,
            # Add these parameters to handle attention masks
            model_kwargs={"use_cache": True, "forced_decoder_ids": None},
            generate_kwargs={"max_length": 448, "return_timestamps": True},  # Whisper's max token length
        )

    def _group_into_sentences(self, chunks):
        sentences = []
        current_sentence = []
        sentence_start = None
        last_end = 0
        SILENCE_THRESHOLD_MS = 500  # Minimum pause to consider as sentence boundary

        for chunk in chunks:
            text = chunk["text"].strip()
            start = chunk["timestamp"][0] * 1000  # Convert to milliseconds
            end = chunk["timestamp"][1] * 1000 if chunk["timestamp"][1] else start + 1000  # Estimate end

            # Detect sentence boundaries using punctuation and silence gaps
            if current_sentence:
                # Check for pause duration between sentences
                silence_gap = start - last_end

                # Check for ending punctuation in previous word
                ends_with_punctuation = any(current_sentence[-1].endswith(p) 
                                          for p in [".", "?", "!", "。", "？", "！"])

                if ends_with_punctuation or silence_gap > SILENCE_THRESHOLD_MS:
                    sentences.append({
                        "text": " ".join(current_sentence),
                        "start": sentence_start,
                        "end": last_end
                    })
                    current_sentence = []
                    sentence_start = None

            if not current_sentence:
                sentence_start = start

            current_sentence.append(text)
            last_end = end

        # Add the final sentence if any remains
        if current_sentence:
            sentences.append({
                "text": " ".join(current_sentence),
                "start": sentence_start,
                "end": last_end
            })

        return sentences

    def transcribe(self):
        """Convert speech to text in original language (Chinese)"""
        if not Settings.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {Settings.audio_file}")

        # Load audio file and get sampling rate
        audio, sampling_rate = librosa.load(
            str(Settings.audio_file), sr=16000, mono=True  # Whisper expects 16kHz  # Force single channel
        )

        # Pass raw audio directly to pipeline
        return self.pipe(
            audio,
            generate_kwargs={
                "language": "zh",
                "task": "transcribe",  # Transcribe only, no translation
                # "task": "translate",
                "forced_decoder_ids": None,
                "return_timestamps": "word",  # Get word-level timestamps
                "use_cache": True,
            },
        )

    def translate(self, sentences: list[dict]) -> list[dict]:
        """Translate Chinese sentences to English using specialist model"""
        translator = Translator()
        chinese_texts = [s["text"] for s in sentences]
        english_texts = translator.translate_batch(chinese_texts)

        for s, en_text in zip(sentences, english_texts):
            # Extract only the English part after the translation prompt
            s["english"] = en_text.strip().split("\n\n")[-1].split(": ")[-1]

        return sentences

    def run(self):
        """Run full audio processing pipeline"""
        # Process through pipeline
        result = self.transcribe()

        # Load audio file to get duration
        if not Settings.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {Settings.audio_file}")
        audio, _ = librosa.load(str(Settings.audio_file), sr=16000, mono=True)

        # Create sentence structure
        chunks = result["chunks"] if "chunks" in result else [{
            "text": result["text"], 
            "timestamp": (0.0, len(audio)/16000)
        }]
        sentences = self._group_into_sentences(chunks)

        # Perform translation
        translated_sentences = self.translate(sentences)

        # Build final results
        json_results = [{
            "original": s["text"],
            "english": s["english"],
            "start_pts": int(s["start"]),
            "end_pts": int(s["end"])
        } for s in translated_sentences]

        # Save results as JSON to output directory
        import json
        with open(Settings.result_audio, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        # return json_results
