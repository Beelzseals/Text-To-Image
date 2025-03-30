import os
import base64
import json
import datetime
import time
from io import BytesIO

from ml_models.model_manager import ModelManager
from models.generation_model import ImageGenerationModel

from PIL import Image
from openai import APITimeoutError, APIConnectionError, RateLimitError, BadRequestError, OpenAIError


# TODO - ADD FLUX MODEL
# TODO - ADD DEEPSEEK MODEL
class ImageGenerator:
    def __init__(self, logger):
        model_manager = ModelManager()
        self.task = "generation"

        self.current_day = datetime.datetime.now().strftime("%Y-%m-%d")
        self.logger = logger
        self.gen_main_path = "images/generation"
        self.flux_folder = "flux"
        self.sd_folder = "stable_diffusion"
        self.dalle_folder = "dall_e"
        # main folders

        self._init_folders()

        prompts_path = "data/prompts.json"
        self.prompt_collection = json.load(open(prompts_path, "r"))

        self.client = model_manager.load_openai_client()
        self.sd_35_medium = model_manager.get_pipeline("generation", "stable_diffusion", "sd_35_medium")
        self.sd_35_large = model_manager.get_pipeline("generation", "stable_diffusion", "sd_35_large")
        self.flux_pipe = None  # Currently not used

    def _init_folders(self):
        """Initializes the folder structure for image generation."""

        os.makedirs("images", exist_ok=True)
        os.makedirs(self.gen_main_path, exist_ok=True)

        os.makedirs(os.path.join(self.gen_main_path, self.sd_folder), exist_ok=True)
        # daily subfolders
        os.makedirs(os.path.join(self.gen_main_path, self.sd_folder, self.current_day), exist_ok=True)
        os.makedirs(os.path.join(self.gen_main_path, self.dalle_folder, self.current_day), exist_ok=True)
        # subfolders for dall-e 2 and 3
        os.makedirs(os.path.join(self.gen_main_path, self.dalle_folder, self.current_day, "dalle_2"), exist_ok=True)
        os.makedirs(os.path.join(self.gen_main_path, self.dalle_folder, self.current_day, "dalle_3"), exist_ok=True)
        # subfolders for stable diffusion
        os.makedirs(os.path.join(self.gen_main_path, self.sd_folder, self.current_day, "sd_35_medium"), exist_ok=True)
        os.makedirs(os.path.join(self.gen_main_path, self.sd_folder, self.current_day, "sd_35_large"), exist_ok=True)

    # =========================================== ERROR HANDLING  ============================================

    def _handle_retryable_error(
        self, error, prompt_entry, image_generation_prompt, prompt_summary, retry_count, max_retries
    ):
        """Handles retryable errors during DALL-E image generation."""
        self.logger.error(f"Retryable error: {error} for prompt summary: {prompt_summary}")

        if retry_count >= max_retries:
            self.logger.error(f"Max retries reached for prompt: {prompt_summary}")
            return

        if isinstance(error, RateLimitError):
            self.logger.info("Rate limit error, waiting for 60 seconds")
            time.sleep(60)

        sanitized_prompt = self._sanitize_prompt(image_generation_prompt)
        prompt_entry["ai_revised"]["positive"].append(sanitized_prompt)
        self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")

        self._generate_dalle_images(prompt_entry, sanitized_prompt, prompt_summary, retry_count + 1, max_retries)

    def _log_and_exit(self, error, prompt_summary):
        """Logs non-retryable errors and exits the program."""
        self.logger.error(f"Non-retryable error: {error} for prompt summary: {prompt_summary}")
        return

    # =========================================== PROMPT HANDLING METHODS ====================================
    def _generate_prompt_entry(self, positive_prompt, summary, negative_prompt=None):
        """Generates a new prompt entry in the prompt collection."""
        # Generate new prompt json if it doesn't exist
        if not os.path.exists("data/prompts.json"):
            self.prompt_collection = {"prompts": []}
        summaries = [prompt["original"]["summary"] for prompt in self.prompt_collection["prompts"]]
        if summary not in summaries:
            entry = {
                "original": {"positive": positive_prompt, "negative": negative_prompt, "summary": summary},
                "ai_revised": {"positive": [], "negative": []},
                "human_revised": {"positive": [], "negative": []},
            }
            self.prompt_collection["prompts"].append(entry)
        else:
            entry = [
                prompt for prompt in self.prompt_collection["prompts"] if prompt["original"]["summary"] == summary
            ][0]
            entry["original"]["positive"] = positive_prompt
            entry["original"]["negative"] = negative_prompt
            entry["original"]["summary"] = summary

        # Save the updated prompt collection to the file
        with open("data/prompts.json", "w") as f:
            json.dump(self.prompt_collection, f, indent=4)
        self.logger.info(f"Prompt entry added for: {summary}")

        return entry

    def _moderate_prompt(self, raw_prompt):
        """Moderates the prompt using OpenAI's moderation API."""
        return self.client.moderations.create(model="text-moderation-stable", input=raw_prompt)

    def _handle_prompt_moderation(self, prompt):
        moderation_res = self._moderate_prompt(prompt)

        if moderation_res.results[0].flagged is False:
            return prompt
        flagged_categories = [category for category, is_flagged in moderation_res.results[0].categories if is_flagged]
        self.logger.info(f"Prompt was flagged for categories: {flagged_categories}")
        return self._sanitize_prompt(prompt, flagged_categories)

    def _sanitize_prompt(self, raw_prompt, flagged_categories=None):
        """ ""Sanitizes the prompt using OpenAI's chat completion API."""
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

    def _refine_prompt_for_retry(self, prompt, retry_count, max_retries, image_generation_prompt):
        if retry_count < max_retries:
            sanitized_prompt = self._sanitize_prompt(image_generation_prompt)
            prompt["ai_revised"]["positive"].append(sanitized_prompt)
            self.logger.info(f"Retrying with sanitized prompt (attempt {retry_count + 1}): {sanitized_prompt}")
            return sanitized_prompt

    # =========================================== HUGGING FACE ====================================

    def _generate_hf_images(self, positive_prompt, neg_prompt, prompt_summary):
        """Generates images using Hugging Face pipelines."""
        # load pipelines
        pipes_data = [
            {"pipe": self.sd_35_medium, "folder": "sd_35_medium"},
            {"pipe": self.sd_35_large, "folder": "sd_35_large"},
        ]

        # generate images
        for pipe_data in pipes_data:
            pipe = pipe_data["pipe"]
            out_path = os.path.join(self.gen_main_path, self.sd_folder, self.current_day, pipe_data["folder"])
            self.logger.info(f"Generating images with {pipe_data['folder']} pipeline")
            images = pipe(
                prompt=positive_prompt,
                negative_prompt=neg_prompt,
                num_images_per_prompt=3,
                guidance_scale=7.5,
                num_inference_steps=28,
            ).images

            for i, img in enumerate(images):
                self._save_hf_images(img, i, prompt_summary, out_path)

    def _save_hf_images(self, img, i, prompt_summary, out_path):
        """Saves the generated images to the specified path."""
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        img_path = f"{out_path}/{prompt_summary}_{time}_{i}.png"
        self.logger.info(f"Saving image: {img_path}")
        img.save(img_path)

    # =========================================== DALL-E 2 & 3 ==============================
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

    def _save_dalle_images(self, res, prompt_summary, model_folder):
        """ ""Saves the generated DALL-E images to the specified path."""
        for i, img_data in enumerate(res.data):
            b64_img = img_data.b64_json
            decoded_img_data = base64.b64decode(b64_img)
            img = Image.open(BytesIO(decoded_img_data))
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            file_name = f"{self.gen_main_path}/{self.dalle_folder}/{self.current_day}/{model_folder}/{prompt_summary}_{time}_{i}.png"
            img.save(file_name)
            self.logger.info(f"Saved image: {file_name}")

    def _generate_dalle_images(self, prompt_entry, positive_prompt, prompt_summary, retry_count=0, max_retries=3):
        moderated_prompt = self._handle_prompt_moderation(positive_prompt)
        print(moderated_prompt)
        try:
            d2_res = self._generate_dalle2_images(moderated_prompt)
            d3_res = self._generate_dalle3_images(moderated_prompt)
            self._save_dalle_images(d2_res, prompt_summary, "dalle_2")
            self._save_dalle_images(d3_res, prompt_summary, "dalle_3")
            self.logger.info(f"Finished generating images for prompt summary: {prompt_summary}")

        except (APITimeoutError, RateLimitError, OpenAIError) as e:
            self._handle_retryable_error(e, prompt_entry, positive_prompt, prompt_summary, retry_count, max_retries)

        except (APIConnectionError, PermissionError, BadRequestError) as e:
            self._log_and_exit(e, prompt_summary)

    def _retrieve_latest_prompts(self, prompt):
        """ ""Retrieves the latest positive and negative prompts from the prompt collection."""
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

    # ============================================== PUBLIC METHODS =================================

    def generate_existing_prompts(self):
        """Generates images for all prompts in the prompt collection, with all available models."""
        for prompt_entry in self.prompt_collection["prompts"]:
            positive_prompt, negative_prompt, prompt_summary = self._retrieve_latest_prompts(prompt_entry)
            self.logger.info(f"Generating images for {prompt_summary}")
            self._generate_hf_images(prompt_entry, positive_prompt, negative_prompt, prompt_summary)
            self._generate_dalle_images(prompt_entry, positive_prompt, prompt_summary)

            # Save back modified prompts
            with open("data/prompts.json", "w") as f:
                json.dump(self.prompt_collection, f, indent=4)

    def generate_with_one_model(self, prompt_data: ImageGenerationModel):
        """Generates a single image for the given prompt using the specified model."""
        model = prompt_data.model
        positive_prompt = prompt_data.positive_prompt
        negative_prompt = prompt_data.negative_prompt
        prompt_summary = prompt_data.prompt_summary
        prompt_entry = self._generate_prompt_entry(positive_prompt, prompt_summary, negative_prompt)

        if model == "DALL-E":
            self._generate_dalle_images(prompt_entry, positive_prompt, prompt_summary)
        elif model == "Stable Diffusion":
            self._generate_hf_images(positive_prompt, negative_prompt, prompt_summary)
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
        prompt_entry = self._generate_prompt_entry(positive_prompt, prompt_summary, negative_prompt)

        self.logger.info(f"Generating images for {prompt_summary}")
        self._generate_hf_images(positive_prompt, negative_prompt, prompt_summary)
        # self.generate_hf_images("flux", positive_prompt, negative_prompt, prompt_summary)
        self._generate_dalle_images(prompt_entry, positive_prompt, prompt_summary)
        self.logger.info(f"Finished generating images for {prompt_summary}")
        # Save back modified prompts
        with open("data/prompts.json", "w") as f:
            json.dump(self.prompt_collection, f, indent=4)

    def get_generation_pipelines(self, model_manager: ModelManager):
        """Loads the generation pipelines."""
        self.sd_35_medium = model_manager.get_pipeline("generation", "stable_diffusion", "sd_35_medium")
        self.sd_35_large = model_manager.get_pipeline("generation", "stable_diffusion", "sd_35_large")
        # self.flux_pipe = model_manager.get_pipeline("generation", "flux", "flux")
