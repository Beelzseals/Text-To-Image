# Text-to-Image

This is a learning project meant to generate images with the help of different AI models belonging to Stable Diffusion and OpenAI 'families'. Current features are limited and there is much planned. 

## Features

- Convert text prompts into high-quality images.
- Support for various styles and resolutions.
- Easy-to-use interface for generating images.

## Installation

1. Clone the repository:

2. Navigate to the project directory

3. Create a virtual environment and activate it

4. Install dependencies

5. Add .env file with api keys to OpenAI and Hugging Face

## Usage

1. Run the application by executing `app.py`:

2. Open your web browser and navigate to `http://localhost:8000` to access the application interface.

3. Enter a text prompt in the input field provided.

4. Select the desired model for the generated image.

5. Click the "Generate" button to create an image based on your text prompt.

6. Once the image is generated you can view it under the images folder.

7. Refer to the `logs/` directory for detailed logs in case of errors or debugging purposes.

## Requirements

- Python 3.8 or higher
- Required libraries listed in `requirements.txt`

## Versioning

### Previous Versions
- **v0.5.0**: Initial release with basic text-to-image generation functionality.
- **v0.8.0**: Add Gradio UI, add Inpainting skeleton

### Upcoming Features
- Finalize inpainting to work
- Add Flux model(s) and Janus for generation
- Further optimize pipelines 
- Introduce advanced style customization options.
- Add better error checking, type infering
