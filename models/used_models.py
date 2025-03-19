import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    FluxPipeline,
    StableDiffusionInpaintPipeline,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    FluxPipeline,
    StableDiffusionInpaintPipeline)
from openai import OpenAI

load_dotenv()
login(token=os.getenv("HF_ACCESS_TOKEN"))



def load_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_sd_model():
    stable_diffusion_model = [["stabilityai/stable-diffusion-3.5-medium", "sd_35_medium"], ["stabilityai/stable-diffusion-3.5-large", "sd_35_large"]]
    pipes = []
    for model in stable_diffusion_model:
        model_path = model[0]
        folder = model[1]
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16
        )

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=model_nf4,
            torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        pipes.append({"pipe": pipe, "folder": folder})

    return pipes


def load_flux_model():
    """CURRENTLY NOT USED"""
    flux_models = ["black-forest-labs/FLUX.1-dev"]
    pipes = []
    for model in flux_models:
        pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.float16)
        lora_repo = "strangerzonehf/Flux-Midjourney-Mix2-LoRA"
        pipe.load_lora_weights(lora_repo)
        # pipe.enable_model_cpu_offload()
        pipe.to(torch.device("cuda"))
        pipes.append(pipe)
    return pipes


def load_inpainter():
    """CURRENTLY NOT USED"""
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe