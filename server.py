from typing import Union
from fastapi import FastAPI
from . import generator

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/generate/flux")
def generate_flux(prompt: str, negative_prompt: str):
    return {"prompt": prompt, "negative_prompt": negative_prompt}