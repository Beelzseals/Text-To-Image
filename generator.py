import os
import requests
from models.used_models import load_sd_model, load_flux_model, load_openai_client
import json
import datetime

flux_folder = "images/Flux"
sd_folder = "images/Stable_diffusion"
dalle_folder = "images/Dall_e"

def init_folders():
    os.makedirs(flux_folder, exist_ok=True)
    os.makedirs(sd_folder, exist_ok=True)
    os.makedirs(dalle_folder, exist_ok=True)
    current_day = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(os.path.join(flux_folder, current_day), exist_ok=True)
    os.makedirs(os.path.join(sd_folder, current_day), exist_ok=True)
    os.makedirs(os.path.join(dalle_folder, current_day), exist_ok=True)


def generate_images(model_type, prompt, neg_prompt, prompt_summary, folder):
    if model_type == "flux":
        pipes = load_flux_model()
        prompt = "MJ v6" + prompt
        neg_prompt = "MJ v6" + neg_prompt
    elif model_type == "sd":
        pipes = load_sd_model()
    images = []

    for pipe in pipes:
        image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=40,
            guidance_scale=7.5,
            num_images_per_prompt=3
        ).images

        for img in image:
            images.append(img)
    save_hf_images(images, prompt_summary, folder)


def dalle_prompt(prompt):
    client = load_openai_client()
    d2_res = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=3
    )
    d3_res = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=4
    )
    save_dalle_images(d2_res["data"], prompt_summary)
    save_dalle_images(d3_res["data"], prompt_summary)


def save_dalle_images(images, prompt_summary):
    for i, img in enumerate(images):
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        img_url = img["url"]
        img_content = requests.get(img_url).content
        file_name = f"../images/Dall_e/{prompt_summary}_{time}_{i}.png"
        with open(file_name, "wb") as f:
            f.write(img_content)


def save_hf_images(sd_images, prompt_summary, folder):
    for i, img in enumerate(sd_images):
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        img.save(f"{folder}/{prompt_summary}_{time}_{i}.png") 


def generate_main():
    init_folders()
    prompts_path = "data/prompts.json"
    data = json.load(open(prompts_path, "r"))
    for prompt in data:
        prompt_summary = prompt["prompt_summary"]
        prompt = prompt["prompt"]
        negative_prompt = prompt["negative_prompt"]
        generate_images("sd", prompt, negative_prompt, prompt_summary, sd_folder)
        #generate_images("flux", prompt, negative_prompt, prompt_summary, flux_folder)
        dalle_prompt(prompt)
