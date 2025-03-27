from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import re


class ImageGenerationModel(BaseModel):
    model: Literal["DALL-E", "Stable Diffusion", "All"] = Field(
        ...,
        description="Model selection for image generation. Choose between DALL-E and Stable Diffusion.",
    )
    positive_prompt: str = Field(
        ...,
        title="Positive Prompt",
        min_length=1,
        description="Elements you want to focus on+ in the generated image",
        max_length=400,
    )
    negative_prompt: Optional[str] = Field(
        None, title="Negative Prompt", description="Elements you want to avoid in the generated image", max_length=400
    )
    prompt_summary: str = Field(
        ...,
        title="Prompt Summary",
        min_length=1,
        description="A short summary of the prompt. Used as the filename for the generated image. Should be in snake case.",
        max_length=20,
        examples=["stray_cat_in_a_forest", "a_red_apple_on_a_table"],
    )

    @field_validator("prompt_summary")
    def validate_prompt_summary(cls, value):
        if not re.match(r"^[a-z0-9_]+$", value):
            raise ValueError("Prompt summary must be in snake case (lowercase letters, numbers, and underscores only).")
        return value
