import importlib
import json

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

from analyzer import BaseAnalyzer
from settings import Settings

OcrModels = {
    "Qwen20": {"id": "Qwen/Qwen2-VL-7B-Instruct", "model": "Qwen2VLForConditionalGeneration"},
    "Qwen25": {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "model": "Qwen2_5_VLForConditionalGeneration"},
}


def import_model(model_dict: dict, name: str):
    entry = model_dict[name]
    module = importlib.import_module("transformers")
    model = getattr(module, entry["model"])
    return model, entry["id"]


qwen_model, qwen_model_id = import_model(OcrModels, "Qwen25")


class OcrAnalyzer(BaseAnalyzer):
    def __init__(self, model_id=qwen_model_id):
        """Initialize the OCR analyzer with Qwen model.

        Args:
            model_id: Model identifier for Qwen
        """
        super().__init__()
        self.model_id = model_id

    def run(self):
        # # default: Load the model on the available device(s)
        # model = qwen_model.from_pretrained(self.model_id, torch_dtype="auto", device_map="auto")

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

        ## Remark
        # flash_attn on Windows is a complete mess. Although I was able to to compile and run it once,
        # it turned out to be quite delicate with regard to the selection of a correct n-tuple of
        # (local LLM (Qwen-VL), CUDA, cuDNN, PyTorch, flash_attn, ...). Any upgrade of any component
        # can destroy the complete functionality by e.g. adding untracked dependencies of similar
        # caliber as the module itself (triton), simply undecipherable error messages and more,
        # including the impossibility to recompile the whole module

        model = qwen_model.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            #    attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # default processer
        processor = AutoProcessor.from_pretrained(self.model_id, response_format={"type": "json_object"})

        # # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256 * 28 * 28
        # max_pixels = 4 * 1280 * 28 * 28
        # processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

        if Settings.result_ocr.exists():
            Settings.result_ocr.unlink()

        images = sorted(list(Settings.out_rois.glob("*.png")))
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
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

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

        with open(Settings.result_ocr, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
