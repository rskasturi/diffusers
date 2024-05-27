from diffusers import StableDiffusionPipeline
import torch
import intel_extension_for_pytorch as ipex
model_id = "./textual_inversion_cat"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("xpu")

prompt = "A backpack with <cat-toy> cat logo"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")