# %%
from datasets import load_dataset
from PIL import Image
import io

dataset = load_dataset("/home/huayu/pickapic_v2",split="validation_unique")

# %%
imageN= 10
prompts = dataset[:imageN]['caption']
raw_images = []
for i in range(imageN):
    jpg_0 = Image.open(io.BytesIO(dataset[i]['jpg_0'])).convert("RGB")
    jpg_1 = Image.open(io.BytesIO(dataset[i]['jpg_1'])).convert("RGB")
    prefered = jpg_0 if dataset[i]['label_0'] > 0.5 else jpg_1
    disprefered = jpg_1 if dataset[i]['label_0'] > 0.5 else jpg_0
    raw_images += [prefered, disprefered]

# %%
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

# %%
generator = torch.Generator(device='cuda')
generator = generator.manual_seed(0)
baseline_images = pipe(prompt=[item for item in prompts for _ in range(2)], generator=generator, guidance_scale=gs).images


# %%
# paths= ['/home/huayu/git/DiffusionDPO/tmp-sd15-fulltest-sft_latest22-0310/checkpoint-500',
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-fulltest-dpo_latest22-0310/checkpoint-500",
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w1/checkpoint-1500",
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w10/checkpoint-1500",
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w100/checkpoint-1500",
#         ]
# paths= ['/home/huayu/git/DiffusionDPO/tmp-sd15-fulltest-sft_latest22-0310/checkpoint-500',
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-fulltest-dpo_latest22-0310/checkpoint-500",
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b001w1000/checkpoint-1000",
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w1000/checkpoint-1000",
#         "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b100w1000/checkpoint-1000",
#         ]
paths= ["/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b001w1000/checkpoint-1000",
        "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b01w1000/checkpoint-1000",
        "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w10000/checkpoint-1000",
        "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b10w1000/checkpoint-1000",
        "/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b100w1000/checkpoint-1000",
        ]
print(paths)
path_images = []
for path in paths:
    dpo_unet = UNet2DConditionModel.from_pretrained(
                                path,
                                # alternatively use local ckptdir (*/checkpoint-n/)
                                subfolder='unet',
                                torch_dtype=torch.float16
    ).to('cuda')
    pipe.unet = dpo_unet

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(0)
    path_images.append(pipe(prompt=[item for item in prompts for _ in range(2)], generator=generator, guidance_scale=gs).images)



# %%
import matplotlib.pyplot as plt
from matplotlib import rcParams
import textwrap


for pid, prompt in enumerate(prompts):
    # 创建一个大图
    fig = plt.figure(figsize=(10, 20))
    fig.suptitle( textwrap.fill(prompt, width=50) )
    prompt_n = 0
    ax = fig.add_subplot(len(paths) + 2, 2, 1)
    ax.imshow(raw_images[pid*2+0])
    ax.axis('off')
    ax.set_title("prefered")


    ax = fig.add_subplot(len(paths) + 2, 2, 2)
    ax.imshow(raw_images[pid*2+1])
    ax.axis('off')
    ax.set_title("disprefered")


    ax = fig.add_subplot(len(paths) + 2, 2, 3)
    ax.imshow(baseline_images[pid*2+0])
    ax.axis('off')
    ax.set_title("sd15")


    ax = fig.add_subplot(len(paths) + 2, 2, 4)
    ax.imshow(baseline_images[pid*2+1])
    ax.axis('off')
    ax.set_title("sd15")

    for i in range(len(paths)):
        ax = fig.add_subplot(len(paths) + 2, 2, 2*i+5 )
        # ax.imshow(path_images[i+ pid* len(paths)][0])
        ax.imshow(path_images[i][0 +  pid* 2])
        ax.axis('off')
        ax.set_title(paths[i][-50:])

        ax = fig.add_subplot(len(paths) + 2, 2, 2*i+6 )
        # ax.imshow(path_images[i+ pid* len(paths)][1])
        ax.imshow(path_images[i][1 +  pid* 2])
        ax.axis('off')
        ax.set_title(paths[i][-50:])


    # 调整子图间距
    plt.tight_layout()
    # 保存图像
    plt.savefig('p{}.png'.format(pid))

    # # 显示图像
    # plt.show()



