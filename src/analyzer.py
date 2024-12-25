import json
import os
from pathlib import Path

# import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from settings import Settings


class Analyzer:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.prompts = self.settings.data_dir / "prompts.json"
        self.roi_dir = self.settings.odir_rois
        with open(self.prompts, "r") as f:
            self.prompts = json.load(f)
        self.ocr_result = self.settings.ocr_result

    def run(self):
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

        # default processer
        processor = AutoProcessor.from_pretrained(model_name, response_format={"type": "json_object"})

        # # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256 * 28 * 28
        # max_pixels = 4 * 1280 * 28 * 28
        # processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

        if self.ocr_result.exists():
            self.ocr_result.unlink()

        images = sorted(list(self.roi_dir.glob("*.png")))
        num_images = len(images)
        chunk_size = 1
        result = []

        for i in range(0, num_images, chunk_size):
            partition = images[i : i + chunk_size]
            files = [item.as_posix() for item in partition]
            prompt = self.prompts["single"]

            content = [{"type": "image", "image": img} for img in files]
            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=5000)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # print(f"  {output_text[0]=}")
            cleaned_text = output_text[0].strip("`").replace("json", "").strip()
            cleaned_text = cleaned_text.replace("\\n", "\n").replace('\\"', '"')

            # Parse the cleaned text as JSON
            try:
                valid_json = json.loads(cleaned_text)
            except json.JSONDecodeError:
                valid_json = None
            result.append(valid_json)

            print(f"{i // chunk_size + 1}/{num_images // chunk_size}: {cleaned_text}")

        with open(self.ocr_result, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
