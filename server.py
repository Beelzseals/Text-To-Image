from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import gradio as gr

# from inpainter import Inpainter
# from generator import ImageGenerator
from utils.cors_config import setup_cors

class PromptModel(BaseModel):
    positive_prompt: str
    negative_prompt: str
    prompt_summary: str



app = FastAPI()
# generator = ImageGenerator()
# inpainter = Inpainter()

setup_cors(app)

def dynamic_ui(task, model=None):
    # Hide all elements initially
    gen_model_visible = inpaint_model_visible = False
    gen_positive_prompt_visible = gen_negative_prompt_visible = gen_prompt_summary_visible = False
    inp_image_visible = inp_mask_visible = inp_prompt_visible = False
    generate_button_visible = False
    
    if task == "Generation":
        gen_model_visible = True
        if model:
            gen_positive_prompt_visible = gen_prompt_summary_visible = True
            generate_button_visible = True
            if model == "DALL-E":
                gen_negative_prompt_visible = True
    elif task == "Inpainting":
        inpaint_model_visible = True
        if model:
            inp_image_visible = inp_mask_visible = inp_prompt_visible = generate_button_visible = True
    
    return (
        gr.update(visible=gen_model_visible),
        gr.update(visible=inpaint_model_visible),
        gr.update(visible=gen_positive_prompt_visible),
        gr.update(visible=gen_negative_prompt_visible),
        gr.update(visible=gen_prompt_summary_visible),
        gr.update(visible=inp_image_visible),
        gr.update(visible=inp_mask_visible),
        gr.update(visible=inp_prompt_visible),
        gr.update(visible=generate_button_visible),
    )


def create_gradio_app():
    with gr.Blocks() as demo:
        task = gr.Radio(["Generation", "Inpainting"], label="Select Task")
        gen_model = gr.Radio(["Stable Diffusion", "DALL-E"], label="Select Model", visible=False)
        inpaint_model = gr.Radio(["Stable Diffusion", "DALL-E"], label="Select Model", visible=False)
        
        gen_positive_prompt = gr.Textbox(label="Positive Prompt", visible=False)
        gen_negative_prompt = gr.Textbox(label="Negative Prompt", visible=False)
        gen_prompt_summary = gr.Textbox(label="Prompt Summary", visible=False)
        
        inp_image = gr.Image(label="Source Image", visible=False, type="pil")
        inp_mask = gr.Image(label="Mask Image", visible=False, type="pil")
        inp_prompt = gr.Textbox(label="Inpainting Prompt", visible=False)
        
        generate_button = gr.Button("Generate", visible=False)
        
        task.change(dynamic_ui, inputs=[task], outputs=[
            gen_model, inpaint_model, gen_positive_prompt, gen_negative_prompt, 
            gen_prompt_summary, inp_image, inp_mask, inp_prompt, generate_button
        ])
        
        gen_model.change(dynamic_ui, inputs=[task, gen_model], outputs=[
            gen_model, inpaint_model, gen_positive_prompt, gen_negative_prompt, 
            gen_prompt_summary, inp_image, inp_mask, inp_prompt, generate_button
        ])
        
        inpaint_model.change(dynamic_ui, inputs=[task, inpaint_model], outputs=[
            gen_model, inpaint_model, gen_positive_prompt, gen_negative_prompt, 
            gen_prompt_summary, inp_image, inp_mask, inp_prompt, generate_button
        ])

    return demo
    



# ==============================ROUTES==============================
# @app.post("/api/generate/dalle")
# def generate_dalle_images(prompt: PromptModel):
#     generator.generate_prompt_entry(prompt.positive_prompt, prompt.negative_prompt, prompt.prompt_summary)
#     generator.generate_dalle_images(prompt)
#     return {"status": "success"}


# @app.post("/api/generate/stable-diffusion")
# def generate_sd_images(prompt: PromptModel):
#     generator.generate_hf_images("sd", prompt)
#     return {"status": "success"}

#NOTE - Uncomment this when Flux model is added
# @app.post("/generate/flux")
# def generate_flux_images(prompt: dict):
#     generator.generate_hf_images("flux", prompt)
#     return {"status": "success"}

# @app.post("/api/generate/all")
# def generate_all_images():
#     generator.generate_all()
#     return {"status": "success"}


# @app.post("/api/inpaint/dalle")
# def inpaint_with_dalle2(image, mask, prompt):
#     return inpainter.inpaint_with_dalle(image, mask, prompt)

# @app.post("/api/inpaint/stable-diffusion")
# def inpaint_with_sd(image, mask, prompt):
#     return inpainter.inpaint_with_sd(image, mask, prompt)




gr_app = create_gradio_app()
app = gr.mount_gradio_app(app, gr_app, "/")

if __name__ == "__main__":
    uvicorn.run("server:app", reload=True)