from math import sqrt
from diffusers.utils import load_image
from diffusers.utils import PIL_INTERPOLATION
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image

from diffsim.hacked_attn import hacked_AttnProcessor2_0, hacked_IPAdapterAttnProcessor2_0
from diffsim.diffsim_pipeline import DiffSimPipeline


def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

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

def sd15_attention_forward_hooked(module, input):
    hidden_states = input[0]
    # print(hidden_states.shape)
    hacked_processor = hacked_AttnProcessor2_0()

    _, query, key, value, _ = hacked_processor(
        module,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None
    )

    module.stores = []
    module.stores = [query, key, value]


def sd15_ipa_attention_forward_hooked(module, input):
    hidden_states, encoder_hidden_states = input[:2]

    hidden_size, cross_attention_dim = module.processor.hidden_size, module.processor.cross_attention_dim
    # hidden_size, cross_attention_dim = hidden_states.shape[-1], 768 # TODO: modify the hard encoding here
    hacked_processor = hacked_IPAdapterAttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, dtype=torch.float16)
    hacked_processor.to_k_ip = module.processor.to_k_ip.to(dtype=torch.float16)
    hacked_processor.to_v_ip = module.processor.to_v_ip.to(dtype=torch.float16)
    hacked_processor = hacked_processor.to(hidden_states.device)
    
    _, query, ip_keys, ip_values, _ = hacked_processor(
        module,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=None
    )

    module.stores = []
    module.stores = [query, ip_keys, ip_values]

class DiffSim:
    def __init__(self, torch_dtype=torch.float16, device='cuda', ip_adapter=False):
        model_id = "ruwnayml/stable-diffusion-v1-5"
        self.pipe = DiffSimPipeline.from_pretrained("/tiamat-NAS/songyiren/models/stable-diffusion-v1-5/", torch_dtype=torch.float16)

        self.device = device
        self.ip_adapter = ip_adapter
        if self.ip_adapter:
            print("Load IP-Adapter")
            self.pipe.load_ip_adapter("/tiamat-NAS/songyiren/Xiaokang/Anti-Reference/ip_adapter/", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
            self.pipe.set_ip_adapter_scale(0.5)
        self.pipe.to(device)

    def prepare_image_latents(self, image, pipe, device, generator=None):
        image = image.to(device=device, dtype=torch.float16)
        ref_image_latents = pipe.vae.encode(image).latent_dist.sample(generator=generator)
        ref_image_latents = pipe.vae.config.scaling_factor * ref_image_latents
        return ref_image_latents
    
    def diffsim(self, image_A, image_B, img_size, prompt, target_block, target_layer, target_step, ip_adapter=False, seed='2333', device='cuda', similarity='cosine'):
        if len(target_layer) == 1:
            target_layer = 0

        # imaeg_A & image_B in the parameter are image path
        A = load_image(image_A)
        B = load_image(image_B)
        tensor_A = process_image(A, img_size)
        tensor_B = process_image(B, img_size)

        # get generator
        generator = get_generator(seed, device)

        # encode image into vae features
        latentsA = self.prepare_image_latents(tensor_A, self.pipe, device, generator)
        latentsB = self.prepare_image_latents(tensor_B, self.pipe, device, generator)

        if ip_adapter:
            ip_A = A
            ip_B = B
        else:
            ip_A = ip_B = None

        # fetch target attention layer
        if target_block == 'down_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.down_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn2.register_forward_pre_hook(sd15_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.down_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn2
            else:
                self.pipe.unet.down_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn1.register_forward_pre_hook(sd15_attention_forward_hooked)
                target_module = self.pipe.unet.down_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn1
        elif target_block == 'mid_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn2.register_forward_pre_hook(sd15_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn2
            else:
                self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn1.register_forward_pre_hook(sd15_attention_forward_hooked)
                target_module = self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn1
        elif target_block == 'up_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.up_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn2.register_forward_pre_hook(sd15_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.up_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn2
            else:
                self.pipe.unet.up_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn1.register_forward_pre_hook(sd15_attention_forward_hooked)
                target_module = self.pipe.unet.up_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn1

        _ = self.pipe.step(prompt=prompt,
                            guidance_scale=7.5,
                            num_inference_steps=1000,
                            target_layer=target_layer,
                            target_block=target_block,
                            generator=generator,
                            latents=latentsA,
                            sample_timestep=target_step,
                            ip_adapter_image=ip_A)

        queryA, keyA, valueA = target_module.stores
        
        _ = self.pipe.step(prompt=prompt,
                            guidance_scale=7.5,
                            num_inference_steps=1000,
                            target_layer=target_layer,
                            target_block=target_block,
                            generator=generator,
                            latents=latentsB,
                            sample_timestep=target_step,
                            ip_adapter_image=ip_B)
        
        queryB, keyB, valueB = target_module.stores

        if ip_adapter:
            attn_a_on_b = [F.scaled_dot_product_attention(queryA, keyB[i], valueB[i], dropout_p=0.0, is_causal=False) for i in range(len(keyB))]
            attn_b_on_a = [F.scaled_dot_product_attention(queryB, keyA[i], valueA[i], dropout_p=0.0, is_causal=False) for i in range(len(keyA))]
            self_attn_a = [F.scaled_dot_product_attention(queryA, keyA[i], valueA[i], dropout_p=0.0, is_causal=False) for i in range(len(keyA))]
            self_attn_b = [F.scaled_dot_product_attention(queryB, keyB[i], valueB[i], dropout_p=0.0, is_causal=False) for i in range(len(keyB))]
        else:
            attn_a_on_b = F.scaled_dot_product_attention(queryA, keyB, valueB, dropout_p=0.0, is_causal=False)
            attn_b_on_a = F.scaled_dot_product_attention(queryB, keyA, valueA, dropout_p=0.0, is_causal=False)
            self_attn_a = F.scaled_dot_product_attention(queryA, keyA, valueA, dropout_p=0.0, is_causal=False)
            self_attn_b = F.scaled_dot_product_attention(queryB, keyB, valueB, dropout_p=0.0, is_causal=False)

        if similarity == 'cosine':
            if ip_adapter:
                diffsim_a_on_b = torch.mean(torch.stack([F.cosine_similarity(a_on_b.reshape(-1).unsqueeze(0), self_a.reshape(-1).unsqueeze(0)) for (a_on_b, self_a) in zip(attn_a_on_b, self_attn_a)], dim=0))
                diffsim_b_on_a = torch.mean(torch.stack([F.cosine_similarity(b_on_a.reshape(-1).unsqueeze(0), self_b.reshape(-1).unsqueeze(0)) for (b_on_a, self_b) in zip(attn_b_on_a, self_attn_b)], dim=0))
            else:
                diffsim_a_on_b = F.cosine_similarity(attn_a_on_b.reshape(-1).unsqueeze(0), self_attn_a.reshape(-1).unsqueeze(0))
                diffsim_b_on_a = F.cosine_similarity(attn_b_on_a.reshape(-1).unsqueeze(0), self_attn_b.reshape(-1).unsqueeze(0))
        else:
            if ip_adapter:
                diffsim_a_on_b = [F.mse_loss(a_on_b, self_a) for (a_on_b, self_a) in zip(attn_a_on_b, self_attn_a)].sum()
                diffsim_b_on_a = [F.mse_loss(b_on_a, self_b) for (b_on_a, self_b) in zip(attn_b_on_a, self_attn_b)].sum()
            else:
                diffsim_a_on_b = F.mse_loss(attn_a_on_b, self_attn_a)
                diffsim_b_on_a = F.mse_loss(attn_b_on_a, self_attn_b)

        return (diffsim_a_on_b + diffsim_b_on_a) / 2


    # only used for extracting QA, QB, QC from a specific image
    def diffsim_value(self, image_A, img_size, prompt, target_block, target_layer, target_step, ip_adapter=False, seed='2333', device='cuda', similarity='cosine'):
        if len(target_layer) == 1:
            target_layer = 0

        # imaeg_A & image_B in the parameter are image path
        A = load_image(image_A)
        tensor_A = process_image(A, img_size)

        # get generator
        generator = get_generator(seed, device)

        # encode image into vae features
        latentsA = self.prepare_image_latents(tensor_A, self.pipe, device, generator)

        if ip_adapter:
            ip_A = A
        else:
            ip_A = ip_B = None

        # fetch target attention layer
        if target_block == 'down_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.down_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn2.register_forward_pre_hook(sd15_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.down_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn2
            else:
                self.pipe.unet.down_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn1.register_forward_pre_hook(sd15_attention_forward_hooked)
                target_module = self.pipe.unet.down_blocks[1:][target_layer].attentions[-1].transformer_blocks[-1].attn1
        elif target_block == 'mid_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn2.register_forward_pre_hook(sd15_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn2
            else:
                self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn1.register_forward_pre_hook(sd15_attention_forward_hooked)
                target_module = self.pipe.unet.mid_block.attentions[-1].transformer_blocks[-1].attn1
        elif target_block == 'up_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.up_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn2.register_forward_pre_hook(sd15_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.up_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn2
            else:
                self.pipe.unet.up_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn1.register_forward_pre_hook(sd15_attention_forward_hooked)
                target_module = self.pipe.unet.up_blocks[:-1][target_layer].attentions[-1].transformer_blocks[-1].attn1

        _ = self.pipe.step(prompt=prompt,
                            guidance_scale=7.5,
                            num_inference_steps=1000,
                            target_layer=target_layer,
                            target_block=target_block,
                            generator=generator,
                            latents=latentsA,
                            sample_timestep=target_step,
                            ip_adapter_image=ip_A)
        
        queryA, keyA, valueA = target_module.stores

        return (queryA, keyA, valueA)


