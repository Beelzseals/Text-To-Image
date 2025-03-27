import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class InpaintModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    inpaint_model: Literal["DALL-E", "Stable Diffusion", "All"] = Field(
        ..., description="Model selection for inpainting"
    )
    inp_image: np.ndarray = Field(..., description="Uploaded image for inpainting")
    inp_mask: np.ndarray = Field(..., description="Mask image defining inpainting areas")
    inp_prompt: str = Field(..., min_length=1, max_length=400, description="Prompt text for inpainting")

    @field_validator("inpaint_model")
    def validate_model(cls, value):
        if value not in ("DALL-E", "Stable Diffusion"):
            raise ValueError("Invalid model selection. Must be 'DALL-E' or 'Stable Diffusion'.")
        return value
