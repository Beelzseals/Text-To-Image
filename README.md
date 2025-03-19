# Text-to-Image

This is a learning project meant to generate images with the help of different AI models belonging to Stable Diffusion and OpenAI 'families'. Current features are limited and there is much planned. 

## Features

- Convert text prompts into high-quality images.
- Support for various styles and resolutions.
- Easy-to-use interface for generating images.

## Installation

1. Clone the repository:

2. Navigate to the project directory:

3. Create a virtual environment and activate it:

4. Install dependencies:

5. Add .env file with api keys to OpenAI and Hugging Face


## Usage

1. Run the application (server.py)
2. Send a post request to "/generate/all" to generate several images with different text-to-image models

## Requirements

- Python 3.8 or higher
- Required libraries listed in `requirements.txt`

## Versioning

This project uses semantic versioning to track features and updates. Below is the version history and planned features:

### Previous Versions
- **v0.5.0**: Initial release with basic text-to-image generation functionality.

### Upcoming Features
- Add support for batch image generation.
- Introduce advanced style customization options.
- Implement a web-based interface for easier accessibility.
- Add inpainting
- Add better error checking, type infering
- Add the option to create new prompts from even a few words
