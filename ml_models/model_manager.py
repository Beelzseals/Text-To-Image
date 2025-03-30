import os
import asyncio

from dotenv import load_dotenv
from huggingface_hub import login
import torch
from openai import OpenAI

from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    StableDiffusionInpaintPipeline,
)


class ModelManager:
    _instance = None
    _initialized = False

    # =============================== MAGIC METHODS ===============================
    def __init__(self):
        if not self._initialized:
            load_dotenv()
            login(token=os.getenv("HF_ACCESS_TOKEN"))
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
            self._openai_client = None
            self._pipeline_cache = {
                "generation": {
                    "stable_diffusion": {
                        "sd_35_medium": None,
                        "sd_35_large": None,
                    },
                    "flux": {
                        # TODO - Add flux model loading here
                        "flux": None,
                    },
                },
                "inpainting": {
                    "stable_diffusion": {
                        "sd_2_inpainting": None,
                    },
                },
            }
            self._initialized = True

    # Singleton pattern to ensure only one instance of ModelManager exists
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    # =============================== PRIVATE METHODS ===============================
    async def _setup_sd_pipeline(self, model_path, folder):
        """Loads a Stable Diffusion pipeline asynchronously."""
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )

        model_nf4 = await asyncio.to_thread(
            SD3Transformer2DModel.from_pretrained,
            model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
        )

        pipe = await asyncio.to_thread(
            StableDiffusion3Pipeline.from_pretrained,
            model_path,
            transformer=model_nf4,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "folder": folder}

    async def _load_sd_model(self):
        """Loads multiple Stable Diffusion pipelines asynchronously."""
        stable_diffusion_model = [
            ["stabilityai/stable-diffusion-3.5-medium", "sd_35_medium"],
            ["stabilityai/stable-diffusion-3.5-large", "sd_35_large"],
        ]
        tasks = [self._setup_sd_pipeline(model[0], model[1]) for model in stable_diffusion_model]
        return await asyncio.gather(*tasks)

    async def _load_inpainter(self):
        """Loads the inpainting model asynchronously."""
        pipe = await asyncio.to_thread(
            StableDiffusionInpaintPipeline.from_pretrained,
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        pipe.to(self.device)
        pipe.enable_attention_slicing()
        return {"pipe": pipe, "folder": "sd_2_inpainting"}

    # =============================== PUBLIC METHODS ===============================
    def load_openai_client(self):
        """Returns a singleton OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client

    async def load_all_pipelines(self):
        """Loads all pipelines asynchronously and caches them for reuse."""

        sd_pipelines_task = self._load_sd_model()
        inpainter_task = self._load_inpainter()

        sd_pipelines, inpainter = await asyncio.gather(sd_pipelines_task, inpainter_task)

        for pipe_info in sd_pipelines:
            pipe = pipe_info["pipe"]
            folder = pipe_info["folder"]
            self._pipeline_cache["generation"]["stable_diffusion"][folder] = pipe
        self._pipeline_cache["inpainting"]["stable_diffusion"][inpainter["folder"]] = inpainter["pipe"]

        return self._pipeline_cache

    def get_pipeline(self, task, model_type, model_name):
        """Returns the requested pipeline from the cache."""
        return self._pipeline_cache[task][model_type][model_name]
