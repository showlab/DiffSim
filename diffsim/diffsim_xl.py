import torch
import torch.nn.functional as F

from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn

from diffsim.diffsim import load_image, process_image, get_generator
from diffsim.diffsim_xl_pipeline import DiffSimXLPipeline
from diffsim.hacked_attn import hacked_AttnProcessor2_0, hacked_IPAdapterAttnProcessor2_0


def sdxl_attention_forward_hooked(module, input):
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

def sdxl_ipa_attention_forward_hooked(module, input):
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


class diffsim_xl:
    def __init__(self, torch_dtype=torch.float16, device='cuda', ip_adapter=False):
        self.pipe = DiffSimXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype, variant="fp16", use_safetensors=True)
        self.device = device
        self.ip_adapter = ip_adapter
        if self.ip_adapter:
            print("Load IP-Adapter")
            self.pipe.load_ip_adapter("/tiamat-NAS/songyiren/Xiaokang/Anti-Reference/ip_adapter/", subfolder="models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
            self.pipe.set_ip_adapter_scale(0.5)
        self.pipe.to(device)

    def prepare_image_latents(self, image, generator=None):
        image = image.to(device=self.device, dtype=torch.float32)
        self.pipe.vae.float()
        ref_image_latents = self.pipe.vae.encode(image).latent_dist.sample(generator=generator)
        ref_image_latents = self.pipe.vae.config.scaling_factor * ref_image_latents
        return ref_image_latents.to(dtype=torch.float16)
    
    def diffsim_score(self, image_A, image_B, img_size, prompt, target_block, target_layer, target_step, similarity, seed):
        # print("Hey")
        A = load_image(image_A)
        B = load_image(image_B)
        tensor_A = process_image(A, img_size)
        tensor_B = process_image(B, img_size)

        # get generator
        generator = get_generator(seed, self.device)

        latentsA = self.prepare_image_latents(tensor_A, generator)
        latentsB = self.prepare_image_latents(tensor_B, generator)

        if self.ip_adapter:
            ip_A = A
            ip_B = B
        else:
            ip_A = ip_B = None

        # fetch target attention layer
        if target_block == 'down_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.down_blocks[1:][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn2.register_forward_pre_hook(sdxl_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.down_blocks[1:][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn2
            else:
                self.pipe.unet.down_blocks[1:][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn1.register_forward_pre_hook(sdxl_attention_forward_hooked)
                target_module = self.pipe.unet.down_blocks[1:][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn1
        elif target_block == 'mid_blocks':
            if self.ip_adapter:
                self.pipe.unet.mid_block.attentions[target_layer[0]].transformer_blocks[target_layer[1]].attn2.register_forward_pre_hook(sdxl_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.mid_block.attentions[target_layer[0]].transformer_blocks[target_layer[1]].attn2
            else:
                self.pipe.unet.mid_block.attentions[target_layer[0]].transformer_blocks[target_layer[1]].attn1.register_forward_pre_hook(sdxl_attention_forward_hooked)
                target_module = self.pipe.unet.mid_block.attentions[target_layer[0]].transformer_blocks[target_layer[1]].attn1
        elif target_block == 'up_blocks':
            # access the target extrating module layer by layer, register for a hooked function
            if self.ip_adapter:
                self.pipe.unet.up_blocks[:-1][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn2.register_forward_pre_hook(sdxl_ipa_attention_forward_hooked)
                target_module = self.pipe.unet.up_blocks[:-1][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn2
            else:
                self.pipe.unet.up_blocks[:-1][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn1.register_forward_pre_hook(sdxl_attention_forward_hooked)
                target_module = self.pipe.unet.up_blocks[:-1][target_layer[0]].attentions[target_layer[1]].transformer_blocks[target_layer[2]].attn1

        self.pipe.step(prompt=prompt,
                       guidance_scale=7.5,
                       num_inference_steps=1000,
                       generator=generator,
                       latents=latentsA,
                       sample_timestep=target_step,
                       ip_adapter_image=ip_A)

        queryA, keyA, valueA = target_module.stores

        self.pipe.step(prompt=prompt,
                       guidance_scale=7.5,
                       num_inference_steps=1000,
                       generator=generator,
                       latents=latentsB,
                       sample_timestep=target_step,
                       ip_adapter_image=ip_B)

        queryB, keyB, valueB = target_module.stores

        if self.ip_adapter:
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
            if self.ip_adapter:
                diffsim_a_on_b = torch.mean(torch.stack([F.cosine_similarity(a_on_b.reshape(-1).unsqueeze(0), self_a.reshape(-1).unsqueeze(0)) for (a_on_b, self_a) in zip(attn_a_on_b, self_attn_a)], dim=0))
                diffsim_b_on_a = torch.mean(torch.stack([F.cosine_similarity(b_on_a.reshape(-1).unsqueeze(0), self_b.reshape(-1).unsqueeze(0)) for (b_on_a, self_b) in zip(attn_b_on_a, self_attn_b)], dim=0))
            else:
                diffsim_a_on_b = F.cosine_similarity(attn_a_on_b.reshape(-1).unsqueeze(0), self_attn_a.reshape(-1).unsqueeze(0))
                diffsim_b_on_a = F.cosine_similarity(attn_b_on_a.reshape(-1).unsqueeze(0), self_attn_b.reshape(-1).unsqueeze(0))
        else:
            if self.ip_adapter:
                diffsim_a_on_b = [F.mse_loss(a_on_b, self_a) for (a_on_b, self_a) in zip(attn_a_on_b, self_attn_a)].sum()
                diffsim_b_on_a = [F.mse_loss(b_on_a, self_b) for (b_on_a, self_b) in zip(attn_b_on_a, self_attn_b)].sum()
            else:
                diffsim_a_on_b = F.mse_loss(attn_a_on_b, self_attn_a)
                diffsim_b_on_a = F.mse_loss(attn_b_on_a, self_attn_b)
        
        return (diffsim_a_on_b + diffsim_b_on_a) / 2

