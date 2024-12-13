import torch
import torch.nn.functional as F

from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMScheduler

from diffsim.diffsim import load_image, process_image, get_generator
from diffsim.diffsim_xl_pipeline import DiffSimXLPipeline
from diffsim.hacked_modules import hacked_CrossAttnUpBlock2D_forward, hacked_CrossAttnDownBlock2D_forward, hacked_UNetMidBlock2DCrossAttn_forward
from diffsim.hacked_attn import hacked_AttnProcessor2_0, hacked_IPAdapterAttnProcessor2_0

from DiT.modelsdit import DiT_models
from DiT.download import find_model
from DiT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL


def dit_attention_forward_hook(module, input):
    x = input[0]
    B, N, C = x.shape
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)

    module.stores = [q, k, v]


class diffsim_DiT:
    def __init__(self, img_size, target_step, device, ckpt=None):
        self.model = DiT_models["DiT-XL/2"](
            input_size=img_size // 8,
            in_channels = 4,
            num_classes=1000
        ).to(device)

        ckpt_path = ckpt or f"DiT-XL-2-{img_size}x{img_size}.pt"
        loaded_state_dict = find_model(ckpt_path)
        model_state_dict = self.model.state_dict()
        new_state_dict = {}
        for key, tensor in loaded_state_dict.items():
            if key in model_state_dict:
                if tensor.size() == model_state_dict[key].size():
                    new_state_dict[key] = tensor
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.half()
        self.model.eval()

        self.diffusion = create_diffusion(str(target_step))
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

        self.scheduler = DDIMScheduler.from_pretrained("/tiamat-NAS/songyiren/models/stable-diffusion-v1-5/", subfolder="scheduler")

        self.device = device

    def prepare_image_latents(self, image, generator=None):
        image = image.to(device=self.device, dtype=torch.float32)
        self.vae.float()
        ref_image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents
        return ref_image_latents.to(dtype=torch.float16)

    def add_noise(self, latents, generator, t):
        noise = randn_tensor(
            latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
        )
        return self.scheduler.add_noise(
            latents,
            noise,
            torch.tensor(t, dtype=torch.int)
            # torch.Tensor([t]),
        )
    
    def diffsim_score(self, image_A, image_B, img_size, prompt, target_block, target_layer, target_step, similarity, seed):
        target_layer = target_layer[0]
        A = load_image(image_A)
        B = load_image(image_B)
        tensor_A = process_image(A, img_size)
        tensor_B = process_image(B, img_size)

        # get generator
        generator = get_generator(seed, self.device)

        latentsA = self.prepare_image_latents(tensor_A, generator)
        latentsB = self.prepare_image_latents(tensor_B, generator)

        latentsA = self.add_noise(latentsA, generator, target_step)
        latentsB = self.add_noise(latentsB, generator, target_step)

        # print(latentsA.shape)
        # latentsA = torch.concat((latentsA, torch.ones()), dim=)

        self.diffusion = create_diffusion(str(target_step))

        y = torch.tensor([1], device=self.device, dtype=torch.int)
        y_null = torch.tensor([1000] * 1, device=self.device, dtype=torch.int)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y)

        self.model.blocks[target_layer].attn.register_forward_pre_hook(dit_attention_forward_hook)

        # _ = self.diffusion.p_sample_loop(
        #     self.model.forward_with_cfg, latentsA.shape, latentsA, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
        # )
        t = torch.tensor([1000 - target_step] * latentsA.shape[0], device=self.device, dtype=torch.int)
        _ = self.diffusion.p_sample(
            self.model,
            latentsA,
            t,
            clip_denoised=False,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
        )

        queryA, keyA, valueA = self.model.blocks[target_layer].attn.stores

        _ = self.diffusion.p_sample(
            self.model,
            latentsB,
            t,
            clip_denoised=False,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
        )

        queryB, keyB, valueB = self.model.blocks[target_layer].attn.stores

        attn_a_on_b = F.scaled_dot_product_attention(queryA, keyB, valueB, dropout_p=0.0, is_causal=False)
        attn_b_on_a = F.scaled_dot_product_attention(queryB, keyA, valueA, dropout_p=0.0, is_causal=False)
        self_attn_a = F.scaled_dot_product_attention(queryA, keyA, valueA, dropout_p=0.0, is_causal=False)
        self_attn_b = F.scaled_dot_product_attention(queryB, keyB, valueB, dropout_p=0.0, is_causal=False)

        if similarity == 'cosine':
            diffsim_a_on_b = F.cosine_similarity(attn_a_on_b.reshape(-1).unsqueeze(0), self_attn_a.reshape(-1).unsqueeze(0))
            diffsim_b_on_a = F.cosine_similarity(attn_b_on_a.reshape(-1).unsqueeze(0), self_attn_b.reshape(-1).unsqueeze(0))
        else:
            diffsim_a_on_b = F.mse_loss(attn_a_on_b, self_attn_a)
            diffsim_b_on_a = F.mse_loss(attn_b_on_a, self_attn_b)
        
        return (diffsim_a_on_b + diffsim_b_on_a) / 2
