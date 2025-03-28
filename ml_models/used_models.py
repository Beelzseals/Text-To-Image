import os
import torch
import asyncio
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    StableDiffusionInpaintPipeline,
)
from openai import OpenAI

# Load environment variables
load_dotenv()
login(token=os.getenv("HF_ACCESS_TOKEN"))

# Define global variables for caching loaded models
_pipeline_cache = None
_openai_client = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_openai_client():
    """Returns a singleton OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def setup_sd_pipeline(model_path, folder):
    """Loads a Stable Diffusion pipeline asynchronously."""
    nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

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


async def load_sd_model():
    """Loads multiple Stable Diffusion pipelines asynchronously."""
    stable_diffusion_model = [
        ["stabilityai/stable-diffusion-3.5-medium", "sd_35_medium"],
        ["stabilityai/stable-diffusion-3.5-large", "sd_35_large"],
    ]
    tasks = [setup_sd_pipeline(model[0], model[1]) for model in stable_diffusion_model]
    return await asyncio.gather(*tasks)


async def load_inpainter():
    """Loads the inpainting model asynchronously."""
    pipe = await asyncio.to_thread(
        StableDiffusionInpaintPipeline.from_pretrained,
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


async def load_all_pipelines():
    """Loads all pipelines asynchronously and caches them for reuse."""
    global _pipeline_cache
    if _pipeline_cache is None:
        sd_pipelines_task = load_sd_model()
        inpainter_task = load_inpainter()

        sd_pipelines, inpainter = await asyncio.gather(sd_pipelines_task, inpainter_task)

        _pipeline_cache = {
            "stable_diffusion_pipelines": sd_pipelines,
            "inpainter_pipeline": inpainter,
        }
    return _pipeline_cache


def get_pipelines():
    """Retrieves cached pipelines synchronously."""
    global _pipeline_cache
    if _pipeline_cache is None:
        raise RuntimeError("Pipelines are not loaded yet. Use asyncio to load them first.")
    return _pipeline_cache
