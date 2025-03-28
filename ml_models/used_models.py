import os
import torch
from dotenv import load_dotenv
import asyncio

from huggingface_hub import login
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    StableDiffusionInpaintPipeline,
)
from openai import OpenAI

load_dotenv()
login(token=os.getenv("HF_ACCESS_TOKEN"))


def load_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def setup_sd_pipeline(model_path, folder):
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
    stable_diffusion_model = [
        ["stabilityai/stable-diffusion-3.5-medium", "sd_35_medium"],
        ["stabilityai/stable-diffusion-3.5-large", "sd_35_large"],
    ]
    tasks = [setup_sd_pipeline(model[0], model[1]) for model in stable_diffusion_model]
    pipes = await asyncio.gather(*tasks)
    return pipes


async def load_inpainter():
    pipe = await asyncio.to_thread(
        StableDiffusionInpaintPipeline.from_pretrained,
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


async def load_all_pipelines():
    # Load all pipelines asynchronously
    sd_pipelines_task = load_sd_model()
    inpainter_task = load_inpainter()

    sd_pipelines, inpainter = await asyncio.gather(sd_pipelines_task, inpainter_task)

    return {
        "stable_diffusion_pipelines": sd_pipelines,
        "inpainter_pipeline": inpainter,
    }
