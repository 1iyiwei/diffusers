import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import platform

print(platform.platform(), flush=True)

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4")

cuda = "cuda"
if torch.backends.mps.is_available():
    cuda = "mps"
else:
    cuda = "cpu"

print(cuda, flush=True)

cuda = 'cpu'
pipe = pipe.to(cuda)

prompt = "a photo of an astronaut riding a horse on mars"
outputFileName = "astronaut_rides_horse.png";

#with autocast(cuda):
    #image = pipe(prompt).images[0]
image = pipe(prompt)["sample"][0]
    
image.save(outputFileName)
