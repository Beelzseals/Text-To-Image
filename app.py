from fastapi import FastAPI
import uvicorn
import gradio as gr

from utils.validation import handle_generation_errors
from utils.validation import handle_inpainting_errors
from inpainter import Inpainter
from generator import ImageGenerator
from utils.cors_config import setup_cors


app = FastAPI()
generator = ImageGenerator()
inpainter = Inpainter()

setup_cors(app)


######################### EVENT HANDLERS ###########################


def handle_generation(*args):
    generation_inputs = handle_generation_errors(*args)
    if generation_inputs.startswith("ERROR"):
        return generation_inputs

    if args[0] == "All":
        generator.generate_with_all_models(generation_inputs)
    else:
        generator.generate_with_one_model(generation_inputs)


def handle_inpainting(*args):
    inpaint_inputs = handle_inpainting_errors(*args)
    if inpaint_inputs.startswith("ERROR"):
        return inpaint_inputs

    if args[0] == "All":
        inpainter.inpaint_with_all_models(inpaint_inputs)
    else:
        inpainter.inpaint_with_one_model(inpaint_inputs)


def setup_generation_event_handlers(
    gen_model: gr.Radio,
    gen_positive_prompt,
    gen_negative_prompt,
    gen_prompt_summary,
    btn_generate_with_selected_model: gr.Button,
    btn_generate_with_all_models: gr.Button,
    gen_error_output: gr.Textbox,
):
    gen_model.change(
        lambda model: gr.update(visible=(model == "DALL-E")),
        inputs=[gen_model],
        outputs=[gen_negative_prompt],
    )

    btn_generate_with_selected_model.click(
        handle_generation,
        inputs=[gen_model, gen_positive_prompt, gen_negative_prompt, gen_prompt_summary],
        outputs=[gen_error_output],
    )

    btn_generate_with_all_models.click(
        handle_generation,
        inputs=[gen_model, gen_positive_prompt, gen_negative_prompt, gen_prompt_summary],
        outputs=[gen_error_output],
    )


def setup_inpainting_event_handlers(
    inpaint_model: gr.Radio,
    inp_image,
    inp_mask,
    inp_prompt,
    inpaint_button,
    inp_error_output,
):
    inpaint_model.change(
        lambda model: gr.update(visible=(model == "DALL-E")),
        inputs=[inpaint_model],
        outputs=[inp_mask],
    )

    inpaint_button.click(
        handle_inpainting,
        inputs=[inpaint_model, inp_image, inp_mask, inp_prompt],
        outputs=[inp_error_output],
    )


# ==============================GRADIO APP==============================
def create_gradio_app():
    with gr.Blocks() as demo:
        with gr.Tab(label="Generation"):
            gen_model = gr.Radio(["DALL-E", "Stable Diffusion", "All"], label="Model")
            gen_positive_prompt = gr.Textbox(lines=3, label="Positive Prompt")
            gen_negative_prompt = gr.Textbox(
                lines=3, label="Negative Prompt", visible=(gen_model == "Stable Diffusion")
            )
            gen_prompt_summary = gr.Textbox(lines=1, label="Prompt Summary")

            gen_error_output = gr.Textbox(lines=2, label="Errors", interactive=False)
            with gr.Row():
                btn_generate_with_selected_model = gr.Button(value="Generate with selected model")
                btn_generate_with_all_models = gr.Button(value="Generate with all models")

                setup_generation_event_handlers(
                    gen_model,
                    gen_positive_prompt,
                    gen_negative_prompt,
                    gen_prompt_summary,
                    btn_generate_with_selected_model,
                    btn_generate_with_all_models,
                    gen_error_output,
                )

        with gr.Tab(label="Inpainting"):
            inpaint_model = gr.Radio(["DALL-E", "Stable Diffusion", "All"], label="Model")
            inp_image = gr.Image(label="Image", sources=["upload", "clipboard"])
            inp_mask = gr.Image(label="Mask", sources=["upload", "clipboard"])
            inp_prompt = gr.Textbox(lines=3, label="Prompt")

            inpaint_button = gr.Button(value="Inpaint")

            inp_error_output = gr.Textbox(lines=2, label="Errors", interactive=False)
            setup_inpainting_event_handlers(
                inpaint_model,
                inp_image,
                inp_mask,
                inp_prompt,
                inpaint_button,
                inp_error_output,
            )

    return demo


# ============================== ROUTES ==================================
@app.post("/api/generate/single")
def generate_single_image():
    return {"status": "success"}


@app.post("/api/generate/all")
def generate_all_images():
    return {"status": "success"}


@app.post("/api/generate/existing-prompts")
def generate_existing_prompts():
    generator.generate_existing_prompts()
    return {"status": "success"}


@app.post("/api/inpaint/single")
def inpaint_image():
    return {"status": "success"}


@app.post("/api/inpaint/all")
def inpaint_all_images():
    return {"status": "success"}


gr_app = create_gradio_app()
app = gr.mount_gradio_app(app, gr_app, "/")

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)
