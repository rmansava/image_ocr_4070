"""Vision model inference via HuggingFace transformers with SDPA."""

from pathlib import Path

import torch
from PIL import Image

DEFAULT_MAX_DIM = 1280

DEFAULT_MODEL = "qwen3-vl-8b"

# Model family registry — maps short names to HF model IDs and classes
MODEL_REGISTRY = {
    "qwen3-vl-8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_class": "Qwen3VLForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
    },
    "qwen3-vl-4b": {
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_class": "Qwen3VLForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
    },
    "qwen2.5-vl-7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
    },
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


class HFVisionEngine:
    """Loads a vision model once and runs inference on images."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dtype: str = "bf16",
        quantize: str | None = None,
        max_dim: int = DEFAULT_MAX_DIM,
        compile_model: bool = True,
    ):
        self.max_dim = max_dim
        self.model_name = model_name
        self.compile_model = compile_model

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list_models()}"
            )

        self.hf_id = MODEL_REGISTRY[model_name]["hf_id"]
        self.model_class = MODEL_REGISTRY[model_name]["model_class"]
        self.processor_class = MODEL_REGISTRY[model_name]["processor_class"]

        # Determine torch dtype
        if dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Quantization config
        self.bnb_config = None
        if quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise RuntimeError(
                    "--quantize 4bit requires bitsandbytes. "
                    "Install with: pip install bitsandbytes>=0.43"
                ) from None
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor into GPU memory."""
        if self.model is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This pipeline requires a GPU."
            )

        import transformers
        model_cls = getattr(transformers, self.model_class)
        proc_cls = getattr(transformers, self.processor_class)

        print(f"Loading {self.hf_id}...")

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }
        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config

        self.model = model_cls.from_pretrained(self.hf_id, **model_kwargs)
        self.processor = proc_cls.from_pretrained(self.hf_id)

        if self.compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile skipped: {e}")

        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        used_gb = total_gb - free_gb
        print(f"Model loaded. VRAM: {used_gb:.1f} / {total_gb:.1f} GB ({free_gb:.1f} GB free)")
        if free_gb < 2.0:
            print(
                f"WARNING: Only {free_gb:.1f} GB VRAM free. Inference may OOM. "
                "Consider --quantize 4bit or --model qwen3-vl-4b."
            )

    def _prepare_image(self, image_path: Path) -> Image.Image:
        """Load and resize image."""
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > self.max_dim:
                ratio = self.max_dim / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            img.load()  # force read before file handle closes
        return img

    def infer(self, image_path: Path, prompt: str) -> str:
        """Run inference on a single image. Returns model output text."""
        self.load()
        img = self._prepare_image(image_path)

        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )

        # Trim input tokens from output
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
