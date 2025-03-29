import io
import base64
from ml_models.model_manager import ModelManager
from PIL import Image
from models.inpaint_model import InpaintModel


class Inpainter:
    def __init__(self):
        model_manager = ModelManager()
        self.sd_pipe = model_manager.get_pipeline("inpainting", "stable_diffusion", "sd_2_inpainting")
        self.flux_pipe = None  # Currently not used
        self.openai_client = model_manager.load_openai_client()

    def _np_to_image(self, np_array):
        return Image.fromarray(np_array)

    def _inpaint_with_sd(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        output = self.sd_pipe(prompt=prompt, image=image, mask_image=mask).images[0]
        return output

    def _inpaint_with_dalle(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")

        # Convert image and mask to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()

        buffered_mask = io.BytesIO()
        mask.save(buffered_mask, format="PNG")
        mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode()

        response = self.openai_client.images.edit(
            image=image_b64, mask=mask_b64, prompt=prompt, n=1, size="512x512", response_format="b64_json"
        )
        result_b64 = response["data"][0]["b64_json"]
        res_img = Image.frombytes("RGB", (512, 512), base64.b64decode(result_b64))

    # def inpaint_with_flux(self,image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
    #     pass

    def inpaint_with_all_models(self, inpaint_inputs: InpaintModel):
        image = self._np_to_image(inpaint_inputs.image)
        mask = self._np_to_image(inpaint_inputs.mask)
        prompt = inpaint_inputs.prompt

        sd_result = self._inpaint_with_sd(image, mask, prompt)
        dalle_result = self._inpaint_with_dalle(image, mask, prompt)

    def inpaint_with_one_model(self, inpaint_inputs: InpaintModel):
        image = self._np_to_image(inpaint_inputs.image)
        mask = self._np_to_image(inpaint_inputs.mask)
        prompt = inpaint_inputs.prompt
        model = inpaint_inputs.model

        if model == "DALL-E":
            result = self._inpaint_with_dalle(image, mask, prompt)
        else:
            result = self._inpaint_with_sd(image, mask, prompt)

    def get_inpainter_pipeline(self, model_manager: ModelManager):
        self.sd_pipe = model_manager.get_pipeline("inpainting", "stable_diffusion", "sd_2_inpainting")
