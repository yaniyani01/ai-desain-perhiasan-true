import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

model_id = "SG161222/Realistic_Vision_V4.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Deskripsi Desain Cincin"),
    outputs=gr.Image(type="pil")
)

demo.launch()
