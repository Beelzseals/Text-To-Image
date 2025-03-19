#TODO - CURRENTLY UNUSED, PLANNED FOR FUTURE VERSION

import io
import base64
import openai
from models.used_models import  load_inpainter
from PIL import Image



def b64_to_image(b64_str):
    image_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGBA")

def image_to_b64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def mask_rgba_to_single_channel(mask):
    return mask.convert("L")


def stable_diffusion_inpaint(image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    pipe = load_inpainter()
    output = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return output


def dalle2_inpaint(image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L")

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()

    buffered_mask = io.BytesIO()
    mask.save(buffered_mask, format="PNG")
    mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode()

    # OpenAI API Call
    response = openai.images.edit(
        image=image_b64,
        mask=mask_b64,
        prompt=prompt,
        n=1,
        size="512x512",
        response_format="b64_json"
    )
    result_b64 = response["data"][0]["b64_json"]
    return b64_to_image(result_b64)
