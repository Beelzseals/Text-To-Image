import os
import base64
import json
import datetime
import time

from io import BytesIO
from models.used_models import load_sd_model, load_flux_model, load_openai_client
from utils.custom_log import create_logger
from PIL import Image
from openai import APITimeoutError, APIConnectionError, RateLimitError, BadRequestError, OpenAIError

# human > ai > original

class ImageGenerator:
    def __init__(self):
        self.flux_folder = "images/Flux"
        self.sd_folder = "images/Stable_diffusion"
        self.dalle_folder = "images/Dall_e"
        self.current_day = datetime.datetime.now().strftime("%Y-%m-%d")
        self.logger = create_logger()
        self._init_folders()
        prompts_path = "data/prompts.json"
        self.prompts = json.load(open(prompts_path, "r"))


    def _init_folders(self):
        # main folders
        os.makedirs(self.flux_folder, exist_ok=True)
        os.makedirs(self.sd_folder, exist_ok=True)
        os.makedirs(self.dalle_folder, exist_ok=True)
        # daily subfolders
        os.makedirs(os.path.join(self.flux_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.sd_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.dalle_folder, self.current_day), exist_ok=True)
        #subfolders for dall-e 2 and 3
        os.makedirs(os.path.join(self.dalle_folder, self.current_day, "dalle_2"), exist_ok=True)
        os.makedirs(os.path.join(self.dalle_folder, self.current_day, "dalle_3"), exist_ok=True)
        # subfolders for stable diffusion
        os.makedirs(os.path.join(self.sd_folder, self.current_day, "sd_35_medium"), exist_ok=True)
        os.makedirs(os.path.join(self.sd_folder, self.current_day, "sd_35_large"), exist_ok=True)


    def generate_images(self, model_type, prompt, positive_prompt, neg_prompt, prompt_summary, folder):
        if model_type == "flux":
            pipes = load_flux_model()
            positive_prompt = "MJ v6" + positive_prompt
            neg_prompt = "MJ v6" + neg_prompt
        elif model_type == "sd":
            pipes = load_sd_model()
        images = []

        for pipe in pipes:
            image = pipe(
                prompt=positive_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=40,
                guidance_scale=7.5,
                num_images_per_prompt=3
            ).images

            for img in image:
                images.append(img)
        self._save_hf_images(images, prompt_summary, folder)


    def _sanitize_prompt(self, raw_prompt, client):
        system_prompt =  "You are an expert at rewriting image generation prompts so they do not violate any content policies, while preserving the intended visual meaning and detail. Avoid sensitive, explicit, or potentially risky wording. Simplify where needed, but keep the main elements clear."
        

        sanitized_prompt_result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": raw_prompt
                }
            ]
        )


        return sanitized_prompt_result.choices[0].message.content


    def _save_dalle_images(self, res, prompt, prompt_summary):
        for i, img_data in enumerate(res.data):
            b64_img = img_data.b64_json
            prompt["result"].append(b64_img)

            decoded_img_data = base64.b64decode(b64_img)
            img = Image.open(BytesIO(decoded_img_data))
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            file_name = f"{self.dalle_folder}/{self.current_day}/{prompt_summary}_{time}_{i}.png"
            img.save(file_name)
            self.logger.info(f"Saved image: {file_name}")


    def _refine_prompt_for_retry(self, prompt, client, retry_count, max_retries, image_generation_prompt):
        if retry_count < max_retries:
                sanitized_prompt = self._sanitize_prompt(image_generation_prompt, client)
                prompt["ai"]["positive"].append(sanitized_prompt)
                self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")
                return sanitized_prompt


    def _generate_dalle2_images(self, client, prompt):
        return client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            response_format="b64_json",
            n=3
        )


    def _generate_dalle3_images(self, client, prompt):
        return client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            style="natural",
            response_format="b64_json",
            n=1
        )
    

    def _handle_retryable_error(self, error, client, prompt, image_generation_prompt, prompt_summary, retry_count, max_retries):
        self.logger.error(f"Retryable error: {error} for prompt summary: {prompt_summary}")

        if retry_count >= max_retries:
            self.logger.error(f"Max retries reached for prompt: {prompt_summary}")
            return

        if isinstance(error, RateLimitError):
            time.sleep(5)

        sanitized_prompt = self._sanitize_prompt(image_generation_prompt, client)
        prompt["ai"]["positive"].append(sanitized_prompt)
        self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")

        self.generate_dalle_images(prompt, sanitized_prompt, prompt_summary, retry_count + 1, max_retries)


    def _log_and_exit(self, error, prompt_summary):
        self.logger.error(f"Non-retryable error: {error} for prompt summary: {prompt_summary}")
        return


    def generate_dalle_images(self,prompt, image_generation_prompt, prompt_summary, retry_count=0, max_retries=3):
        client = load_openai_client()
        try:
            d2_res = self._generate_dalle2_images(client, image_generation_prompt)
            d3_res = self._generate_dalle3_images(client, image_generation_prompt)
            time.sleep(2)
            self._save_dalle_images(d2_res,prompt, prompt_summary)
            self._save_dalle_images(d3_res,prompt, prompt_summary)
            self.logger.info(f"Finished generating images for prompt summary: {prompt_summary}")

        except (APITimeoutError, RateLimitError, OpenAIError) as e:
            self._handle_retryable_error(e, client, prompt, image_generation_prompt, prompt_summary, retry_count, max_retries)

        except (APIConnectionError, PermissionError, BadRequestError) as e:
            self._log_and_exit(e, prompt_summary)


    def _save_hf_images(self, sd_images, prompt_summary, folder):
        for i, img in enumerate(sd_images):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            img_path = f"{folder}/{self.current_day}/{prompt_summary}_{time}_{i}.png"
            self.logger.info(f"Saving image: {img_path}")
            img.save(img_path)


    def _define_prompts(self, prompt):
        positive_prompt = ""
        negative_prompt = ""
        prompt_summary = prompt["original"]["summary"]

        # finding the latest positive prompt
        if len(prompt["human_revised"]["positive"]) > 0:
            positive_prompt = prompt["human_revised"]["positive"][-1]
        elif len(prompt["ai_revised"]["positive"]) > 0:
            positive_prompt = prompt["ai_revised"]["positive"][-1]
        else:
            positive_prompt = prompt["original"]["positive"]

        # finding the latest negative prompt
        if len(prompt["human_revised"]["negative"]) > 0:
            negative_prompt = prompt["human_revised"]["negative"][-1]
        elif len(prompt["ai_revised"]["negative"]) > 0:
            negative_prompt = prompt["ai_revised"]["negative"][-1]
        else:
            negative_prompt = prompt["original"]["negative"]

        return positive_prompt, negative_prompt, prompt_summary


    def generate_main(self):
        for prompt in self.prompts["prompts"]:
            positive_prompt, negative_prompt, prompt_summary =  self._define_prompts(self, prompt)
            self.logger.info(f"Generating images for {prompt_summary}")
            self.generate_images("sd", prompt, positive_prompt, negative_prompt, prompt_summary, self.sd_folder)
            # self.generate_images("flux", prompt, positive_prompt, negative_prompt, prompt_summary, self.flux_folder)
            self.generate_dalle_images(prompt, positive_prompt, prompt_summary)

            # Save back modified prompts
            with open("data/prompts.json", "w") as f:
                json.dump(self.prompts, f, indent=4)
            


if __name__ == "__main__":
    generator = ImageGenerator()
    generator.generate_main()
