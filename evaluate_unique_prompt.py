from datasets import load_dataset
from PIL import Image
import io
import os
import numpy as np
import argparse

dataset = load_dataset("/home/huayu/pickapic_v2",split="validation_unique")
# Can do clip_utils, aes_utils, hps_utils
from utils.pickscore_utils import Selector
# Score generations automatically w/ reward model
ps_selector = Selector('cuda')
prompts = dataset['caption']
assert len(prompts) == 500
# prefered_imgs = []
# disprefered_imgs = []
# for i in range(500):
#     jpg_0 = Image.open(io.BytesIO(dataset[i]['jpg_0'])).convert("RGB")
#     jpg_1 = Image.open(io.BytesIO(dataset[i]['jpg_1'])).convert("RGB")
#     prefered = jpg_0 if dataset[i]['label_0'] > 0.5 else jpg_1
#     disprefered = jpg_1 if dataset[i]['label_0'] > 0.5 else jpg_0
#     prefered_imgs.append(prefered)
#     disprefered_imgs.append(disprefered)
    


from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
import torch
# pretrained_model_name = "CompVis/stable-diffusion-v1-4"
pretrained_model_name = "/home/huayu/stable-diffusion-v1-5"
# pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
gs = (5 if 'stable-diffusion-xl' in pretrained_model_name else 7.5)
torch.set_grad_enabled(False)
if 'stable-diffusion-xl' in pretrained_model_name:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                   torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.safety_checker = None # Trigger-happy, blacks out >50% of "robot tiger"


#baselines 

baseline_images = []
baseline_scores = []
for i in range(500):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(0)
    img = pipe(prompt=prompts[i*1:(i+1)*1], generator=generator, guidance_scale=gs).images
    score = ps_selector.score(img,prompts[i*1:(i+1)*1])
    baseline_images += img
    baseline_scores += score

    

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--path", type=str, default="None"
)
args = parser.parse_args()

dpo_unet = UNet2DConditionModel.from_pretrained(
                            args.path,
                            # alternatively use local ckptdir (*/checkpoint-n/)
                            subfolder='unet',
                            torch_dtype=torch.float16
).to('cuda')

pipe.unet = dpo_unet
images = []
scores = []
for i in range(500):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(0)
    img = pipe(prompt=prompts[i*1:(i+1)*1], generator=generator, guidance_scale=gs).images
    score = ps_selector.score(img,prompts[i*1:(i+1)*1])
    images += img
    scores += score
    
    
    
scores = np.array(scores)
baseline_scores = np.array(baseline_scores)

print("Path in {}".format(args.path))
print("Win rate v.s. base1.5 is {}".format(np.mean(scores > baseline_scores)))
print("Baseline score is {}".format(np.mean(baseline_scores)))
print("Our score is {}".format(np.mean(scores)))
print("absolute score difference is  {}".format(np.mean(np.abs(baseline_scores - scores))))
