import os
import requests
from openai import BadRequestError
from models.used_models import load_sd_model, load_flux_model, load_openai_client
from utils.custom_log import create_logger
import json
import datetime

flux_folder = "images/Flux"
sd_folder = "images/Stable_diffusion"
dalle_folder = "images/Dall_e"
current_day = datetime.datetime.now().strftime("%Y-%m-%d")
logger = create_logger()

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

def sanitize_prompt(prompt, client):
    system_prompt = (
        "You are an expert at rewriting image generation prompts so they do not violate "
        "any content policies, while preserving the intended visual meaning and detail. "
        "Avoid sensitive, explicit, or potentially risky wording. Simplify where needed, "
        "but keep the main elements clear."
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
)

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def generate_dalle_images(prompt, prompt_summary):
    client = load_openai_client()
    try:
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
            style="natural",
            n=1
        )
    except BadRequestError as e:
        logger.error(f"Error: {e} for prompt: {prompt_summary}")
        sanitized_prompt = sanitize_prompt(prompt, client)
        generate_dalle_images(sanitized_prompt, prompt_summary)
   
    d2_revised_prompts = [data["revised_prompts"] for data in d2_res["data"] if data["revised_prompts"] is not None]
    d2_request_ids = d2_res["_request_ids"]
    logger.info(f"Request IDs: {d2_request_ids}")
    logger.info(f"Revised Prompts: {d2_revised_prompts}") if d2_revised_prompts else None
    for img in d2_res["data"]:
        save_dalle_images(img["url"], prompt_summary)
    
    d3_revised_prompt = d3_res["data"][0]["revised_prompt"]
    d3_request_id = d3_res["_request_id"]
    logger.info(f"Request ID: {d3_request_id}")
    logger.info(f"Revised Prompt: {d3_revised_prompt}") if d3_revised_prompt is not None else None
    save_dalle_images(d3_res["data"], prompt_summary)


def save_dalle_images(images, prompt_summary):
    for i, img in enumerate(images):
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        img_url = img["url"]
        img_content = requests.get(img_url).content
        file_name = f"../images/Dall_e/{current_day}/{prompt_summary}_{time}_{i}.png"
        with open(file_name, "wb") as f:
            f.write(img_content)


def save_hf_images(sd_images, prompt_summary, folder):
    for i, img in enumerate(sd_images):
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        img.save(f"{folder}/{current_day}/{prompt_summary}_{time}_{i}.png") 


def generate_main():
    init_folders()
    prompts_path = "data/prompts.json"
    data = json.load(open(prompts_path, "r"))
    for prompt in data["prompts"]:
        prompt_summary = prompt["original"]["summary"]
        positive_prompt = prompt["original"]["positive"]
        negative_prompt = prompt["original"]["negative"]
        print(f"Generating images for {prompt_summary}")
        logger.info(f"Generating images for {prompt_summary}")
        # generate_images("sd", positive_prompt, negative_prompt, prompt_summary, sd_folder)
        #generate_images("flux", positive_prompt, negative_prompt, prompt_summary, flux_folder)
        generate_dalle_images(positive_prompt, prompt_summary)

generate_main()