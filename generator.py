import os
import base64
import json
import datetime
import time

from io import BytesIO
from ml_models.used_models import load_sd_model, load_openai_client
from utils.custom_log import create_logger
from PIL import Image
from openai import APITimeoutError, APIConnectionError, RateLimitError, BadRequestError, OpenAIError
from models.generation_model import ImageGenerationModel


# TODO - OPTIMIZE FLUX MODEL TO WORK
class ImageGenerator:
    def __init__(self):
        self.flux_folder = "images/Flux"
        self.sd_folder = "images/Stable_diffusion"
        self.dalle_folder = "images/Dall_e"
        self.current_day = datetime.datetime.now().strftime("%Y-%m-%d")
        self.logger = create_logger()
        self._init_folders()
        prompts_path = "data/prompts.json"
        self.prompt_collection = json.load(open(prompts_path, "r"))
        self.client = load_openai_client()

    def _init_folders(self):
        # main folders
        # os.makedirs(self.flux_folder, exist_ok=True)
        os.makedirs(self.sd_folder, exist_ok=True)
        os.makedirs(self.dalle_folder, exist_ok=True)
        # daily subfolders
        # os.makedirs(os.path.join(self.flux_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.sd_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.dalle_folder, self.current_day), exist_ok=True)
        # subfolders for dall-e 2 and 3
        os.makedirs(os.path.join(self.dalle_folder, self.current_day, "dalle_2"), exist_ok=True)
        os.makedirs(os.path.join(self.dalle_folder, self.current_day, "dalle_3"), exist_ok=True)
        # subfolders for stable diffusion
        os.makedirs(os.path.join(self.sd_folder, self.current_day, "sd_35_medium"), exist_ok=True)
        os.makedirs(os.path.join(self.sd_folder, self.current_day, "sd_35_large"), exist_ok=True)

    def _generate_prompt_entry(self, positive_prompt, summary, negative_prompt=None):
        """Generates a new prompt entry in the prompt collection."""
        if not os.path.exists("data/prompts.json"):
            self.prompt_collection = {"prompts": []}
        entry = {
            "original": {"positive": positive_prompt, "negative": negative_prompt, "summary": summary},
            "ai_revised": {"positive": [], "negative": []},
            "human_revised": {"positive": [], "negative": []},
        }
        self.prompt_collection["prompts"].append(entry)
        with open("data/prompts.json", "w") as f:
            json.dump(self.prompt_collection, f, indent=4)
        self.logger.info(f"Prompt entry added for: {summary}")
        return entry

    def _moderate_prompt(self, raw_prompt):
        return self.client.moderations.create(model="text-moderation-stable", input=raw_prompt)

    def _handle_prompt_moderation(self, prompt):
        moderation_res = self._moderate_prompt(prompt)

        if moderation_res.results[0].flagged is False:
            return prompt
        flagged_categories = [category for category, is_flagged in moderation_res.results[0].categories if is_flagged]
        self.logger.info(f"Prompt was flagged for categories: {flagged_categories}")
        return self._sanitize_prompt(prompt, flagged_categories)

    def _generate_hf_images(self, model_type, positive_prompt, neg_prompt, prompt_summary):
        folder = ""

        # load pipelines
        if model_type == "flux":
            folder = self.flux_folder
            # pipes_data = load_flux_model()
            positive_prompt = "MJ v6" + positive_prompt
            neg_prompt = "MJ v6" + neg_prompt
        elif model_type == "sd":
            folder = self.sd_folder
            pipes_data = load_sd_model()

        # generate images
        for pipe_data in pipes_data:
            pipe = pipe_data["pipe"]
            model_folder = pipe_data["folder"]
            images = pipe(
                prompt=positive_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=40,
                guidance_scale=7.5,
                num_images_per_prompt=3,
            ).images

            for i, img in enumerate(images):
                self._save_hf_images(img, i, prompt_summary, folder, model_folder)

    def _sanitize_prompt(self, raw_prompt, flagged_categories=None):
        system_prompt = (
            "You are an expert at rewriting image generation prompts so they do not violate any content policies, while preserving the intended visual meaning and detail. Avoid sensitive, explicit, or potentially risky wording. Simplify where needed, but keep the main elements clear."
            if flagged_categories is None
            else f"Your prompt was flagged for the following categories: {flagged_categories}. Please rewrite it in a way that avoids these categories. Simplify where needed, but keep the main elements clear."
        )

        sanitized_prompt_result = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_prompt}],
        )
        return sanitized_prompt_result.choices[0].message.content

    def _save_dalle_images(self, res, prompt_summary, model_folder):
        for i, img_data in enumerate(res.data):
            b64_img = img_data.b64_json
            decoded_img_data = base64.b64decode(b64_img)
            img = Image.open(BytesIO(decoded_img_data))
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            file_name = f"{self.dalle_folder}/{self.current_day}/{model_folder}/{prompt_summary}_{time}_{i}.png"
            img.save(file_name)
            self.logger.info(f"Saved image: {file_name}")

    def _refine_prompt_for_retry(self, prompt, retry_count, max_retries, image_generation_prompt):
        if retry_count < max_retries:
            sanitized_prompt = self._sanitize_prompt(image_generation_prompt)
            prompt["ai_revised"]["positive"].append(sanitized_prompt)
            self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")
            return sanitized_prompt

    def _generate_dalle2_images(self, prompt):
        return self.client.images.generate(
            model="dall-e-2", prompt=prompt, size="1024x1024", quality="hd", response_format="b64_json", n=3
        )

    def _generate_dalle3_images(self, prompt):
        return self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            style="natural",
            response_format="b64_json",
            n=1,
        )

    def _handle_retryable_error(self, error, prompt, image_generation_prompt, prompt_summary, retry_count, max_retries):
        self.logger.error(f"Retryable error: {error} for prompt summary: {prompt_summary}")

        if retry_count >= max_retries:
            self.logger.error(f"Max retries reached for prompt: {prompt_summary}")
            return

        if isinstance(error, RateLimitError):
            self.logger.info("Rate limit error, waiting for 60 seconds")
            time.sleep(60)

        sanitized_prompt = self._sanitize_prompt(image_generation_prompt)
        prompt["ai_revised"]["positive"].append(sanitized_prompt)
        self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")

        self._generate_dalle_images(prompt, sanitized_prompt, prompt_summary, retry_count + 1, max_retries)

    def _log_and_exit(self, error, prompt_summary):
        self.logger.error(f"Non-retryable error: {error} for prompt summary: {prompt_summary}")
        return

    def _generate_dalle_images(self, prompt, image_generation_prompt, prompt_summary, retry_count=0, max_retries=3):
        moderated_prompt = self._handle_prompt_moderation(image_generation_prompt)
        print(moderated_prompt)
        try:
            d2_res = self._generate_dalle2_images(moderated_prompt)
            d3_res = self._generate_dalle3_images(moderated_prompt)
            self._save_dalle_images(d2_res, prompt_summary, "dalle_2")
            self._save_dalle_images(d3_res, prompt_summary, "dalle_3")
            self.logger.info(f"Finished generating images for prompt summary: {prompt_summary}")

        except (APITimeoutError, RateLimitError, OpenAIError) as e:
            self._handle_retryable_error(e, prompt, image_generation_prompt, prompt_summary, retry_count, max_retries)

        except (APIConnectionError, PermissionError, BadRequestError) as e:
            self._log_and_exit(e, prompt_summary)

    def _save_hf_images(self, img, i, prompt_summary, folder, model_foler):
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        img_path = f"{folder}/{self.current_day}/{model_foler}/{prompt_summary}_{time}_{i}.png"
        self.logger.info(f"Saving image: {img_path}")
        img.save(img_path)

    def _retrieve_latest_prompts(self, prompt):
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

    def generate_existing_prompts(self):
        """Generates images for all prompts in the prompt collection, with all available models."""
        for prompt in self.prompt_collection["prompts"]:
            positive_prompt, negative_prompt, prompt_summary = self._retrieve_latest_prompts(prompt)
            self.logger.info(f"Generating images for {prompt_summary}")
            self._generate_hf_images("sd", prompt, positive_prompt, negative_prompt, prompt_summary)
            # self.generate_hf_images("flux", prompt, positive_prompt, negative_prompt, prompt_summary)
            self._generate_dalle_images(prompt, positive_prompt, prompt_summary)

            # Save back modified prompts
            with open("data/prompts.json", "w") as f:
                json.dump(self.prompt_collection, f, indent=4)

    def generate_with_one_model(self, prompt_data: ImageGenerationModel):
        """Generates a single image for the given prompt using the specified model."""
        model = prompt_data.model
        positive_prompt = prompt_data.positive_prompt
        prompt_summary = prompt_data.prompt_summary
        self._generate_prompt_entry(positive_prompt, prompt_summary)

        if model == "DALL-E":
            self._generate_dalle_images(prompt_data, positive_prompt, prompt_summary)
        elif model == "Stable Diffusion":
            negative_prompt = prompt_data.negative_prompt
            self._generate_hf_images("sd", prompt_data, positive_prompt, negative_prompt, prompt_summary)
        elif model == "Flux":
            # self.generate_hf_images("flux", prompt_data, positive_prompt, negative_prompt, prompt_summary)
            pass
        with open("data/prompts.json", "w") as f:
            json.dump(self.prompt_collection, f, indent=4)

    def generate_with_all_models(self, prompt_data: ImageGenerationModel):
        """Generates images for all prompts in the prompt collection, with all available models."""
        positive_prompt = prompt_data.positive_prompt
        negative_prompt = prompt_data.negative_prompt
        prompt_summary = prompt_data.prompt_summary
        self._generate_prompt_entry(positive_prompt, prompt_summary, negative_prompt)

        self.logger.info(f"Generating images for {prompt_summary}")
        self._generate_hf_images("sd", prompt_data, positive_prompt, negative_prompt, prompt_summary)
        # self.generate_hf_images("flux", prompt_data, positive_prompt, negative_prompt, prompt_summary)
        self._generate_dalle_images(prompt_data, positive_prompt, prompt_summary)
        self.logger.info(f"Finished generating images for {prompt_summary}")
        # Save back modified prompts
        with open("data/prompts.json", "w") as f:
            json.dump(self.prompt_collection, f, indent=4)


# if __name__ == "__main__":
#     generator = ImageGenerator()
#     generator.generate_all()
