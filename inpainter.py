import io
import base64
from models.used_models import  load_inpainter, load_openai_client
from PIL import Image

class Inpainter:
    def __init__(self):
        self.sd_pipe = load_inpainter()
        self.openai_client = load_openai_client()

    def _b64_to_image(self,b64_str):
        image_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGBA")

    def _image_to_b64(self,img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _mask_rgba_to_single_channel(self,mask):
        return mask.convert("L")


    def inpaint_with_sd(self,image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        output = self.sd_pipe(prompt=prompt, image=image, mask_image=mask).images[0]
        return output


    def inpaint_with_dalle(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
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
            image=image_b64,
            mask=mask_b64,
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="b64_json"
        )
        result_b64 = response["data"][0]["b64_json"]
        return self._b64_to_image(result_b64)

    # def inpaint_with_flux(self,image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
    #     pass
