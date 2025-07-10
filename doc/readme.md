## Download/Conversion

### General

- yt-dlp --all-subs https://www.youtube.com/watch?v=GmrH691SEoA    (19)
- https://www.youtube.com/watch?v=MInZHueuLDQ   (20)
- https://www.youtube.com/watch?v=p7ueTuIyQH0   (21)
- https://www.youtube.com/watch?v=qr11Wv5MUOA   (22)

### Audio
- ffmpeg -i .\AiO=ep19.mkv -vn -acodec flac AiO-ep19.flac


## OCR
- [Performing OCR Task with Claude 3 Haiku](https://cevo.com.au/post/performing-ocr-task-with-claude-3-haiku-part-1/)
- [Extracting Data From PDFs Using AI: Claude 3, Donut, and Nougat](https://parsio.io/blog/extracting-data-from-pdfs-using-ai-claude-3-donut-and-nougat/) 


## Audio
  - [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
  - [Qwen](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)

### Whisper
  - [GitHub](https://github.com/openai/whisper)
  - [HuggingFace](https://huggingface.co/openai/whisper-large-v3)
  - [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo) (same model: faster, slightly degraded quality)
  - [Enhancing Whisper transcriptions (OpenAI)](https://cookbook.openai.com/examples/whisper_processing_guide)
  - [whisperx](https://github.com/m-bain/whisperX)
    - depends on pyannote (needs HuggingFace token for diarization))
  - [whisper-timestamped ](https://github.com/linto-ai/whisper-timestamped)  
   
#### API Reference
  - [kind of...](https://deepinfra.com/openai/whisper-large/api?example=openai-speech-http)
  - [DecodingOptions](https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/decoding.py#L81)
  - [HF pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)

### Translation
  - [Qwen2.5-7B-Instruct ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  

### Segmentation
  - [Audio Segmentation into Sentence](https://github.com/openai/whisper/discussions/1243)
  - [spaCy ](https://spacy.io/)

### Diarization
  - [Diarization](https://github.com/lablab-ai/Whisper-transcription_and_diarization-speaker-identification-)
  - [Pyannote vs. NeMo](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)
  - [Top Speaker Diarization Libraries and APIs in 2024](https://www.assemblyai.com/blog/top-speaker-diarization-libraries-and-apis/)
  
