from models.generation_model import ImageGenerationModel
from models.inpaint_model import InpaintModel


def handle_generation_errors(*args):
    try:
        generation_inputs = ImageGenerationModel(*args)
        return generation_inputs
    except Exception as e:
        return "ERROR" + str(e)


def handle_inpainting_errors(*args):
    try:
        inpaint_inputs = InpaintModel(*args)
        return inpaint_inputs
    except Exception as e:
        return "ERROR" + str(e)
