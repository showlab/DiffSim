import argparse
from math import sqrt
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn
from diffusers.utils import PIL_INTERPOLATION
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image

from diffsim.hacked_modules import hacked_CrossAttnUpBlock2D_forward, hacked_CrossAttnDownBlock2D_forward, hacked_UNetMidBlock2DCrossAttn_forward
from diffsim.diffsim_pipeline import DiffSimPipeline
from metrics.hooks import diffusion_self_attention_forward_hook


def arg_parse():
    parser = argparse.ArgumentParser(description="Parse some command-line arguments.")
    parser.add_argument('--image_A', type=str, help='Image A Path')
    parser.add_argument('--image_B', type=str, help='Image B Path')
    parser.add_argument('--target_block', type=str, choices=['down_blocks', 'mid_blocks', 'up_blocks'], default='up_blocks', help='Where the target layer lies within')
    parser.add_argument('--target_layer', type=int, default=2, help='The No. of target layer to calculate the metric')
    parser.add_argument('--target_step', type=int, default=100, help='The target denoising timestep to calculate the metric')
    parser.add_argument('--seed', type=int, default=2333, help='Seed')

    return parser.parse_args()

def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

def plot_heat_map(tensor):
    # print(tensor.shape)
    a = tensor[0]
    size = int(sqrt(a.shape[0]))
    a = a.view(size, size, -1).norm(dim=-1).softmax(dim=-1)
    # a = a / torch.norm(a, p=2)
    # a = transforms.Resize((512, 512))(a.unsqueeze(0).unsqueeze(0))[0][0]
    a = a.cpu().numpy()
    ax = sns.heatmap(a)
    plt.savefig('vis/heatmap.png')

def process_image(image_, img_size=512):
    image_ = image_.convert("RGB")
    image_ = image_.resize((img_size, img_size), resample=PIL_INTERPOLATION["lanczos"])
    image_ = np.array(image_)
    image_ = image_[None, :]

    image = [image_]

    image = np.concatenate(image, axis=0)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5 # vae pixel range: [-1, 1]
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    return image

def prepare_image_latents(image, pipe, device, generator=None):
    image = image.to(device=device, dtype=torch.float16)
    ref_image_latents = pipe.vae.encode(image).latent_dist.sample(generator=generator)
    ref_image_latents = pipe.vae.config.scaling_factor * ref_image_latents
    return ref_image_latents

def mask_query(Q, mask, device):
    if mask is None:
        return Q
    # Q: [1, head, H * W, dim]
    latent_size = int(sqrt(Q.shape[2]))
    mask = torch.from_numpy(mask).to(device, dtype=Q.dtype)
    dilated_mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2) # dilate the mask to ensure major information is preserved
    latent_mask = F.interpolate(dilated_mask, size=(latent_size, latent_size), mode='bilinear', align_corners=False).flatten(2, 3).unsqueeze(3) # [1, 1, latent_size^2, 1]
    # test = F.interpolate(dilated_mask, size=(latent_size, latent_size), mode='bilinear', align_corners=False).cpu().numpy()[0][0]
    # image_array = (test * 255).astype(np.uint8)
    # image = Image.fromarray(image_array, mode='L') # 'L' mode for grayscale
    # image.save('latent_mask_t.png')
    return Q * latent_mask

def cross_attn(hidden_states, attn2, attn1, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs, norm_type, norm2, pos_embed, residual, hidden_states_residual):
    # linear proj
    hidden_states = attn1.to_out[0](hidden_states)
    # dropout
    hidden_states = attn1.to_out[1](hidden_states)

    if attn1.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn1.rescale_output_factor

    hidden_states = hidden_states + hidden_states_residual

    if norm_type == "ada_norm":
        # norm_hidden_states = norm2(hidden_states, timestep)
        pass
    elif norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = norm2(hidden_states)
    elif norm_type == "ada_norm_single":
        # For PixArt norm2 isn't applied here:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
        norm_hidden_states = hidden_states
    elif norm_type == "ada_norm_continuous":
        # norm_hidden_states = norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        pass
    else:
        raise ValueError("Incorrect norm")

    if pos_embed is not None and norm_type != "ada_norm_single":
        norm_hidden_states = pos_embed(norm_hidden_states)

    attn_output = attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
        **cross_attention_kwargs,
    )

    return attn_output

def attention_calc(Q, K, V):
    d_k = K.shape[-1]  # Dimension of the key vectors
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float16))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def diffeats(image_A, image_B, img_size, diffusion_pipe, prompt, target_block, target_layer, target_step, seed='2333', device='cuda', similarity='cosine'):
    if len(target_layer) == 1:
        target_layer = target_layer[0]

    # imaeg_A & image_B in the parameter are image path
    A = load_image(image_A)
    B = load_image(image_B)
    tensor_A = process_image(A, img_size)
    tensor_B = process_image(B, img_size)

    # get generator
    generator = get_generator(seed, device)

    # fetch target attention layer
    if target_block == 'down_blocks':
        target_blocks = diffusion_pipe.unet.down_blocks[:-1] # filter the normal CNN DownBlock2D layer
        forward_func = hacked_CrossAttnDownBlock2D_forward
        module_class = CrossAttnDownBlock2D
    elif target_block == 'mid_blocks':
        target_blocks = [diffusion_pipe.unet.mid_block]
        forward_func = hacked_UNetMidBlock2DCrossAttn_forward
        module_class = UNetMidBlock2DCrossAttn
    elif target_block == 'up_blocks':
        target_blocks = diffusion_pipe.unet.up_blocks[1:] # filter the normal CNN UpBlock2D layer
        forward_func = hacked_CrossAttnUpBlock2D_forward
        module_class = CrossAttnUpBlock2D
    
    # replace with hacked forward function
    # use a hook here
    target_blocks[target_layer].attentions[-1].transformer_blocks[-1].attn1.register_forward_hook(diffusion_self_attention_forward_hook)
    # target_blocks[target_layer].forward = forward_func.__get__(target_blocks[target_layer], module_class)

    # encode image into vae features
    latentsA = prepare_image_latents(tensor_A, diffusion_pipe, device, generator)
    latentsB = prepare_image_latents(tensor_B, diffusion_pipe, device, generator)

    _ = diffusion_pipe.step(prompt=prompt,
                                  guidance_scale=7.5,
                                  num_inference_steps=1000,
                                  target_layer=target_layer,
                                  target_block=target_block,
                                  generator=generator,
                                  latents=latentsA,
                                  sample_timestep=target_step,
                                  diffeats=True)

    featsA = target_blocks[target_layer].attentions[-1].transformer_blocks[-1].attn1.stores[0]

    _ = diffusion_pipe.step(prompt=prompt,
                                  guidance_scale=7.5,
                                  num_inference_steps=1000,
                                  target_layer=target_layer,
                                  target_block=target_block,
                                  generator=generator,
                                  latents=latentsB,
                                  sample_timestep=target_step,
                                  diffeats=True)

    featsB = target_blocks[target_layer].attentions[-1].transformer_blocks[-1].attn1.stores[0]

    featsA = min_max_normalize(featsA)
    featsB = min_max_normalize(featsB)
    
    return F.cosine_similarity(featsA.reshape(-1).unsqueeze(0), featsB.reshape(-1).unsqueeze(0))



if __name__ == '__main__':
    args = arg_parse()
    device = 'cuda'

    # load image
    imageA = load_image(args.image_A)
    imageB = load_image(args.image_B)

    # init sd model
    prompt = "High quality image"
    model_id = "ruwnayml/stable-diffusion-v1-5"
    pipe = DiffSimPipeline.from_pretrained("/tiamat-NAS/songyiren/models/stable-diffusion-v1-5/", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    diffsim_value = diffeats(imageA, imageB, pipe, prompt, args.target_block, args.target_layer, args.target_step, args.seed, device)
    
    print("DiffSim value =", diffsim_value.item())
