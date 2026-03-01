"""Vision model inference via HuggingFace transformers with SDPA."""

from pathlib import Path

import torch
from PIL import Image

DEFAULT_MAX_DIM = 1280

# Model family registry — maps short names to HF model IDs and loader functions
MODEL_REGISTRY = {
    "qwen3-vl-8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen-vl",
    },
    "qwen3-vl-4b": {
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "family": "qwen-vl",
    },
    "qwen2.5-vl-7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen-vl",
    },
    "internvl3.5-8b": {
        "hf_id": "OpenGVLab/InternVL3_5-8B",
        "family": "internvl",
    },
    "internvl3.5-4b": {
        "hf_id": "OpenGVLab/InternVL3_5-4B",
        "family": "internvl",
    },
    "florence-2-large": {
        "hf_id": "microsoft/Florence-2-large",
        "family": "florence",
    },
    "minicpm-v-2.6": {
        "hf_id": "openbmb/MiniCPM-V-2_6",
        "family": "minicpm",
    },
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


class HFVisionEngine:
    """Loads a vision model once and runs inference on images."""

    def __init__(
        self,
        model_name: str = "qwen3-vl-8b",
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

        info = MODEL_REGISTRY[model_name]
        self.hf_id = info["hf_id"]
        self.family = info["family"]

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
            from transformers import BitsAndBytesConfig
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

        print(f"Loading {self.hf_id} ({self.family})...")

        if self.family == "qwen-vl":
            self._load_qwen_vl()
        elif self.family == "florence":
            self._load_florence()
        elif self.family == "internvl":
            self._load_internvl()
        elif self.family == "minicpm":
            self._load_minicpm()

        if self.compile_model and self.family not in ("internvl", "minicpm"):
            # torch.compile for fused kernels — skip for trust_remote_code models
            # that use custom forward() methods incompatible with compile
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile skipped: {e}")

        print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    def _load_qwen_vl(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # Qwen3-VL uses same class as Qwen2.5-VL in transformers
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }
        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_id, **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)

    def _load_florence(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=self.torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=True
        )

    def _load_internvl(self):
        from transformers import AutoModel, AutoTokenizer

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config

        self.model = AutoModel.from_pretrained(self.hf_id, **model_kwargs)
        self.processor = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True
        )

    def _load_minicpm(self):
        from transformers import AutoModel, AutoTokenizer

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }
        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config

        self.model = AutoModel.from_pretrained(self.hf_id, **model_kwargs)
        self.processor = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True
        )

    def _prepare_image(self, image_path: Path) -> Image.Image:
        """Load and resize image."""
        img = Image.open(image_path)
        if img.mode not in ("RGB",):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > self.max_dim:
            ratio = self.max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    def infer(self, image_path: Path, prompt: str) -> str:
        """Run inference on a single image. Returns model output text."""
        self.load()
        img = self._prepare_image(image_path)

        if self.family == "qwen-vl":
            return self._infer_qwen_vl(img, prompt)
        elif self.family == "florence":
            return self._infer_florence(img, prompt)
        elif self.family == "internvl":
            return self._infer_internvl(img, prompt)
        elif self.family == "minicpm":
            return self._infer_minicpm(img, prompt)

        raise RuntimeError(f"No inference method for family: {self.family}")

    def _infer_qwen_vl(self, img: Image.Image, prompt: str) -> str:
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

    def _infer_florence(self, img: Image.Image, prompt: str) -> str:
        # Florence uses task prompts like <MORE_DETAILED_CAPTION>
        task = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task, images=img, return_tensors="pt").to(
            self.model.device, self.torch_dtype
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=1024, num_beams=3
            )

        result = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result, task=task, image_size=(img.width, img.height)
        )
        return parsed.get(task, result).strip()

    def _infer_internvl(self, img: Image.Image, prompt: str) -> str:
        # InternVL uses its own chat interface
        pixel_values = self._internvl_load_image(img).to(
            self.model.device, self.torch_dtype
        )
        generation_config = {"max_new_tokens": 1024, "do_sample": False}

        response = self.model.chat(
            self.processor, pixel_values, prompt,
            generation_config, history=None, return_history=False
        )
        return response.strip()

    def _internvl_load_image(self, img: Image.Image) -> torch.Tensor:
        """Process image for InternVL models."""
        from torchvision import transforms

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(img).unsqueeze(0)

    def _infer_minicpm(self, img: Image.Image, prompt: str) -> str:
        msgs = [{"role": "user", "content": [img, prompt]}]
        result = self.model.chat(image=None, msgs=msgs, tokenizer=self.processor)
        return result.strip()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
