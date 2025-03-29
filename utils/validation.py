from models.generation_model import ImageGenerationModel
from models.inpaint_model import InpaintModel


def handle_generation_errors(gen_model, gen_positive_prompt, gen_negative_prompt, gen_prompt_summary):
    try:
        generation_inputs = ImageGenerationModel(
            model=gen_model,
            positive_prompt=gen_positive_prompt,
            negative_prompt=gen_negative_prompt,
            prompt_summary=gen_prompt_summary,
        )
        return generation_inputs
    except Exception as e:
        return "ERROR" + str(e)


def handle_inpainting_errors(model, image, mask, prompt):
    try:
        inpaint_inputs = InpaintModel(
            model=model,
            image=image,
            mask=mask,
            prompt=prompt,
        )
        return inpaint_inputs
    except Exception as e:
        return "ERROR" + str(e)
