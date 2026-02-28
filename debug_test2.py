"""Debug: see raw model output."""
import transformers.integrations.fsdp as _fsdp_module
_fsdp_module.is_fsdp_managed_module = lambda *args, **kwargs: False

import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

model_name = "deepseek-ai/DeepSeek-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)

# Try the official prompt format
prompts = [
    "<image>\n<|grounding|>Convert the document to markdown.",
    "<image>\nFree OCR.",
    "<image>\nExtract all text from this image.",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print('='*60)
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file="test_images/clear_comic.png",
        output_path="C:/Users/rick/image_ocr/test_output",
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=False,
        eval_mode=True,
    )
    print(f"RESULT TYPE: {type(result)}")
    print(f"RESULT: '{result}'")
