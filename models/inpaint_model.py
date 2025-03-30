from pydantic import BaseModel, Field, field_validator
import numpy as np

from typing import Literal


class InpaintModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    model: Literal["DALL-E", "Stable Diffusion", "All"] = Field(..., description="Model selection for inpainting")
    image: np.ndarray = Field(..., description="Uploaded image for inpainting")
    mask: np.ndarray = Field(..., description="Mask image defining inpainting areas")
    prompt: str = Field(..., min_length=1, max_length=400, description="Prompt text for inpainting")

    @field_validator("model")
    def validate_model(cls, value):
        if value not in ("DALL-E", "Stable Diffusion", "All"):
            raise ValueError("Invalid model selection. Must be 'DALL-E' or 'Stable Diffusion'.")
        return value
