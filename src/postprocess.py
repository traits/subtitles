import json
from enum import IntFlag
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import Settings


class ProcessType(IntFlag):
    """Type of processing being performed"""

    NONE = 0
    OCR = 1
    AUDIO = 2
    BOTH = OCR | AUDIO  # Combined processing type


class PostProcessor:
    def __init__(self):
        # Add stream-specific output files
        basename = Settings.media_base_name
        self.sub_files = {
            "ocr": Settings.out_dir / f"{basename}_ocr.srt",
            "audio": Settings.out_dir / f"{basename}_audio.srt",
            "combined": Settings.out_dir / f"{basename}.srt",
        }

        # Add Qwen model for OCR deduplication
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.dedup_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            sliding_window=-1,  # Disable sliding window attention
        )
        self.dedup_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            padding_side="left",
        )

        # Load prompts
        with open(Settings.root / "data" / "prompts.json", encoding="utf8") as f:
            self.prompts = json.load(f)

    def run(self, process_type: ProcessType = ProcessType.OCR):
        """Run subtitle file generation for the specified processing type.

        Args:
            process_type: Type of processing to generate subtitles for (OCR, AUDIO, or BOTH)
        """
        if process_type == ProcessType.NONE:
            return

        # Handle OCR processing
        if process_type & ProcessType.OCR:
            self.writeOcrSubFile()

        # Handle Audio processing
        if process_type & ProcessType.AUDIO:
            self.writeAudioSubFile()

        # Only create combined file if both OCR and Audio are processed
        if process_type & ProcessType.BOTH == ProcessType.BOTH:
            self.writeCombinedSubFile()

    def mergeSubTitleInfo(self) -> list:
        with open(Settings.result_ocr, "r", encoding="utf8") as f:
            sinfo = json.load(f)
        with open(Settings.log_frame_info, "r") as f:
            finfo = json.load(f)

        ls = len(sinfo)
        lf = len(finfo)
        if ls != lf:
            raise ValueError(f"Data length mismatch: OCR results ({ls} entries) don't match frame info ({lf} entries)")

        result = []
        for si, fi in zip(sinfo, finfo):
            elem = si
            if elem is None:
                elem = {}

            elem["pts"] = fi["in_pts"]
            elem["frame"] = fi["frame"]
            result.append(elem)

        return result

    def _millisecs_to_srt_time(self, ms):
        """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
        ms = max(ms, 0)  # Prevent negative values
        hours = ms // 3_600_000
        ms %= 3_600_000
        minutes = ms // 60_000
        ms %= 60_000
        seconds = ms // 1_000
        ms %= 1_000
        return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

    def _write_subtitle_file(self, file_key: str, subtitle_data: list):
        """Helper method to write subtitle data to a file using buffered writes.

        Args:
            file_key: Key in self.sub_files dictionary for the output file
            subtitle_data: List of subtitle entries to write
        """
        buffer = []
        subtitle_index = 1
        last_english = ""
        last_i = len(subtitle_data) - 1

        for i, v in enumerate(subtitle_data):
            if i < last_i:
                if (text := v.get("english")) and text != last_english:
                    # Handle both OCR (pts) and audio (start_pts/end_pts) formats
                    start_time = self._millisecs_to_srt_time(v.get("start_pts", v.get("pts")))
                    next_entry = subtitle_data[i + 1]
                    end_time = self._millisecs_to_srt_time(next_entry.get("start_pts", next_entry.get("pts")))

                    buffer.append(f"{subtitle_index}\n")
                    buffer.append(f"{start_time} --> {end_time}\n")
                    buffer.append(f"{text}\n\n")
                    subtitle_index += 1
                    last_english = text

                    # Write buffer in chunks
                    if len(buffer) >= 100:  # Write every 100 subtitles
                        with open(self.sub_files[file_key], "a", encoding="utf8") as f:
                            f.writelines(buffer)
                        buffer = []

        # Write remaining buffer
        if buffer:
            with open(self.sub_files[file_key], "a", encoding="utf8") as f:
                f.writelines(buffer)

    def _deduplicate_ocr_texts(self, ocr_info: list) -> list:
        """Merge consecutive OCR entries with minor text variations using batch processing."""
        if not ocr_info:
            return []

        processed = []
        batch_size = 32
        pairs = []
        indices = []

        # Create pairs of consecutive entries
        for i in range(len(ocr_info) - 1):
            current = ocr_info[i]
            next_entry = ocr_info[i + 1]

            if current.get("english") and next_entry.get("english"):
                pairs.append((current, next_entry))
                indices.append(i)

        # Process in batches
        for batch_start in range(0, len(pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(pairs))
            batch_pairs = pairs[batch_start:batch_end]

            print(f"{batch_start}/{len(pairs)}")

            # Prepare batch prompts
            prompts = [
                self.prompts["ocr_deduplication"].format(text1=pair[0]["english"], text2=pair[1]["english"])
                for pair in batch_pairs
            ]

            # Tokenize batch
            inputs = self.dedup_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)

            # Generate responses
            outputs = self.dedup_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0,
                eos_token_id=self.dedup_tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0],
                pad_token_id=self.dedup_tokenizer.eos_token_id,
                do_sample=False,
            )

            # Process responses
            responses = self.dedup_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for idx, response in zip(range(batch_start, batch_end), responses):
                current = pairs[idx][0]
                next_entry = pairs[idx][1]

                # Merge if response isn't "no" and contains valid text
                if "no" not in response.lower() and len(response) > 0:
                    # Use latest timestamp from current entry
                    current["english"] = response.split(":")[-1].strip()
                    current["pts"] = next_entry["pts"]
                    processed.append(current)
                else:
                    processed.append(current)
                    processed.append(next_entry)

        # Handle remaining entries
        if len(processed) < len(ocr_info):
            processed.extend(ocr_info[len(processed) :])

        return processed

    def writeOcrSubFile(self):
        """Create subtitle file from OCR analysis results in SRT format (timestamp based)."""
        info = self.mergeSubTitleInfo()
        info = self._deduplicate_ocr_texts(info)  # Add deduplication step
        self._write_subtitle_file("ocr", info)

    def writeAudioSubFile(self):
        """Create subtitle file from audio analysis results in SRT format (timestamp based)."""
        with open(Settings.result_audio, "r", encoding="utf8") as f:
            audio_info = json.load(f)

        # Filter out duplicate consecutive entries
        filtered_audio = []
        last_english = ""
        for entry in audio_info:
            if entry.get("english") != last_english:
                filtered_audio.append(entry)
                last_english = entry.get("english")

        self._write_subtitle_file("audio", filtered_audio)

    def writeCombinedSubFile(self):
        """Create a combined subtitle file with both OCR and audio streams."""

        def write_subtitle_stream(f, stream_type: str, subtitle_data: list, color_code: str):
            """Nested function to write a subtitle stream.

            Args:
                f: File handle to write to
                stream_type: Type of stream (OCR or AUDIO)
                subtitle_data: List of subtitle entries
                color_code: HTML color code for the stream
            """
            nonlocal subtitle_index
            last_i = len(subtitle_data) - 1
            for i, v in enumerate(subtitle_data):
                if i < last_i:
                    if text := v.get("english"):
                        # Handle both OCR (pts) and audio (start_pts) formats
                        start_time = self._millisecs_to_srt_time(v.get("start_pts", v.get("pts")))
                        next_entry = subtitle_data[i + 1]
                        end_time = self._millisecs_to_srt_time(next_entry.get("start_pts", next_entry.get("pts")))

                        f.write(f"{subtitle_index}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f'<font color="{color_code}">[{stream_type}] {text}</font>\n\n')
                        subtitle_index += 1

        # Load OCR data
        ocr_info = self.mergeSubTitleInfo()
        # Load audio data
        with open(Settings.result_audio, "r", encoding="utf8") as f:
            audio_info = json.load(f)

        with open(self.sub_files["combined"], "w", encoding="utf8") as f:
            subtitle_index = 1

            # Write OCR stream
            write_subtitle_stream(f, "OCR", ocr_info, "#00FF00")

            # Write Audio stream
            write_subtitle_stream(f, "Audio", audio_info, "#FFFF00")
