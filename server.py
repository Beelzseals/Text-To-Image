from fastapi import FastAPI
from generator import ImageGenerator
import uvicorn

app = FastAPI()
generator = ImageGenerator()

@app.post("/generate/dalle")
def generate_dalle_images(prompt: dict):

    generator.generate_dalle_images(prompt)
    return {"status": "success"}

@app.post("/generate/stable-diffusion")
def generate_sd_images(prompt: dict):
    generator.generate_hf_images("flux", prompt)
    return {"status": "success"}

@app.post("/generate/flux")
def generate_flux_images(prompt: dict):
    generator.generate_hf_images(prompt)
    return {"status": "success"}

@app.post("/generate/all")
def generate_all_images():
    generator.generate_all()
    return {"status": "success"}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)