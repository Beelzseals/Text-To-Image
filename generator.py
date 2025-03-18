import os
import base64
import json
import datetime

from io import BytesIO
from models.used_models import load_sd_model, load_flux_model, load_openai_client
from utils.custom_log import create_logger
from PIL import Image


class ImageGenerator:
    def __init__(self):
        self.flux_folder = "images/Flux"
        self.sd_folder = "images/Stable_diffusion"
        self.dalle_folder = "images/Dall_e"
        self.current_day = datetime.datetime.now().strftime("%Y-%m-%d")
        self.logger = create_logger()
        self.init_folders()

    def init_folders(self):
        os.makedirs(self.flux_folder, exist_ok=True)
        os.makedirs(self.sd_folder, exist_ok=True)
        os.makedirs(self.dalle_folder, exist_ok=True)
        os.makedirs(os.path.join(self.flux_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.sd_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.dalle_folder, self.current_day), exist_ok=True)


    def generate_images(self, model_type, prompt, neg_prompt, prompt_summary, folder):
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
        self.save_hf_images(images, prompt_summary, folder)


    def sanitize_prompt(self, prompt, client):
        system_prompt =  "You are an expert at rewriting image generation prompts so they do not violate any content policies, while preserving the intended visual meaning and detail. Avoid sensitive, explicit, or potentially risky wording. Simplify where needed, but keep the main elements clear."
        

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


    def save_dalle_images(self, res, prompt_summary):
        for i, img_data in enumerate(res.data):
            b64_img = img_data.b64_json
            decoded_img_data = base64.b64decode(b64_img)
            img = Image.open(BytesIO(decoded_img_data))
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            file_name = f"{self.dalle_folder}/{self.current_day}/{prompt_summary}_{time}_{i}.png"
            img.save(file_name)
            self.logger.info(f"Saved image: {file_name}")


    def generate_dalle_images(self, prompt, prompt_summary, retry_count=0, max_retries=3):
        client = load_openai_client()
        try:
            # DALL-E 2 generation
            d2_res = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                response_format="b64_json",
                n=3
            )
            # DALL-E 3 generation
            d3_res = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                style="natural",
                response_format="b64_json",
                n=1
            )

        except Exception as e:
            self.logger.error(f"Error: {e} for prompt summary: {prompt_summary}")
            if retry_count < max_retries:
                sanitized_prompt = self.sanitize_prompt(prompt, client)
                self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")
                return self.generate_dalle_images(sanitized_prompt, prompt_summary, retry_count + 1, max_retries)
            else:
                self.logger.error(f"Max retries reached for prompt: {prompt_summary}")
                return

        finally:
            self.save_dalle_images(d2_res, prompt_summary)
            self.save_dalle_images(d3_res, prompt_summary)
            self.logger.info(f"Finished generating images for prompt summary: {prompt_summary}")


    def save_hf_images(self, sd_images, prompt_summary, folder):
        for i, img in enumerate(sd_images):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            img.save(f"{folder}/{self.current_day}/{prompt_summary}_{time}_{i}.png")

    def generate_main(self):
        prompts_path = "data/prompts.json"
        data = json.load(open(prompts_path, "r"))
        for prompt in data["prompts"]:
            prompt_summary = prompt["original"]["summary"]
            positive_prompt = prompt["original"]["positive"]
            negative_prompt = prompt["original"]["negative"]
            print(f"Generating images for {prompt_summary}")
            self.logger.info(f"Generating images for {prompt_summary}")
            # self.generate_images("sd", positive_prompt, negative_prompt, prompt_summary, self.sd_folder)
            # self.generate_images("flux", positive_prompt, negative_prompt, prompt_summary, self.flux_folder)
            self.generate_dalle_images(positive_prompt, prompt_summary)


if __name__ == "__main__":
    generator = ImageGenerator()
    generator.generate_main()
