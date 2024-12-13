import math
import os
from typing import Literal, TypeAlias

import fire
import megfile
import torch
import torch.nn as nn
import torch.nn.functional as F
# from accelerate import PartialState
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
import PIL
import numpy as np

from metrics.hooks import clip_encoder_layer_forward_hook, clip_encoder_layer_feature_forward_hook

class IdentityDict(dict):

    def __missing__(self, key):
        if key is None:
            return None
        return key
    
MODEL_ZOOS = IdentityDict(
    {
        "huggingface/model_name_or_path": "path/to/snapshots",
        # ...
    }
)
_DEFAULT_MODEL: str = MODEL_ZOOS["openai/clip-vit-base-patch32"]
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32

ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor

class CLIPScore:

    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = CLIPModel.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_or_name_path, local_files_only=local_files_only)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def get_text_features(self, text: str | list[str], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
        inputs = self.processor(text=text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(
            inputs["input_ids"].to(self.device),
            inputs["attention_mask"].to(self.device),
        )
        if norm:
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device, dtype=self.dtype))
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def clipi_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        images1_features = self.get_image_features(images1, norm=True)
        images2_features = self.get_image_features(images2, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (images1_features * images2_features).sum(axis=-1)
        return score.sum(0).float(), len(images1)

    @torch.no_grad()
    def clipt_score(self, texts: str | list[str], images: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(images, list):
            images = [images]
        assert len(texts) == len(images), f"Number of texts ({len(texts)}) and images {(len(images))} should be same."

        texts_features = self.get_text_features(texts, norm=True)
        images_features = self.get_image_features(images, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (texts_features * images_features).sum(axis=-1)
        return score.sum(0).float(), len(texts)
    
    @torch.no_grad()
    def attention_calc(self, q, k, v, dropout, training, scale, hidden_size_shape, out_proj):
        bsz, tgt_len, embed_dim = hidden_size_shape
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout if training else 0.0,
            scale=scale,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = out_proj(attn_output)
        return attn_output

    @torch.no_grad()
    def clip_cross_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType], target_layer: int):
        if len(target_layer) == 1:
            target_layer = target_layer[0]
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."
        
        # register the hook (which saves q, k, v in module.store)
        # clip vit total 12 layers, each using CLIPSdpaAttention
        dropout, training, scale = self.model.vision_model.encoder.layers[target_layer].self_attn.dropout, self.model.vision_model.encoder.layers[target_layer].self_attn.training, self.model.vision_model.encoder.layers[target_layer].self_attn.scale
        out_proj = self.model.vision_model.encoder.layers[target_layer].self_attn.out_proj
        self.model.vision_model.encoder.layers[target_layer].register_forward_hook(clip_encoder_layer_forward_hook)

        _ = self.get_image_features(images1, norm=True)
        queryA, keyA, valueA, hidden_size_shape = self.model.vision_model.encoder.layers[target_layer].stores

        _ = self.get_image_features(images2, norm=True)
        queryB, keyB, valueB, hidden_size_shape = self.model.vision_model.encoder.layers[target_layer].stores

        attn_a_on_b = self.attention_calc(queryA, keyB, valueB, dropout, training, scale, hidden_size_shape, out_proj)
        attn_b_on_a = self.attention_calc(queryB, keyA, valueA, dropout, training, scale, hidden_size_shape, out_proj)
        self_attn_a = self.attention_calc(queryA, keyA, valueA, dropout, training, scale, hidden_size_shape, out_proj)
        self_attn_b = self.attention_calc(queryB, keyB, valueB, dropout, training, scale, hidden_size_shape, out_proj)

        diffsim_a_on_b = F.cosine_similarity(attn_a_on_b.reshape(-1).unsqueeze(0), self_attn_a.reshape(-1).unsqueeze(0))
        diffsim_b_on_a = F.cosine_similarity(attn_b_on_a.reshape(-1).unsqueeze(0), self_attn_b.reshape(-1).unsqueeze(0))

        return (diffsim_a_on_b + diffsim_b_on_a) / 2

    @torch.no_grad()
    def clip_feature_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType], target_layer: int):
        if len(target_layer) == 1:
            target_layer = target_layer[0]
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."
        
        # register the hook (which saves q, k, v in module.store)
        # clip vit total 12 layers, each using CLIPSdpaAttention
        # dropout, training, scale = self.model.vision_model.encoder.layers[target_layer].self_attn.dropout, self.model.vision_model.encoder.layers[target_layer].self_attn.training, self.model.vision_model.encoder.layers[target_layer].self_attn.scale
        # out_proj = self.model.vision_model.encoder.layers[target_layer].self_attn.out_proj
        self.model.vision_model.encoder.layers[target_layer].register_forward_hook(clip_encoder_layer_feature_forward_hook)

        _ = self.get_image_features(images1, norm=True)
        attn_maps_A = self.model.vision_model.encoder.layers[target_layer].stores[0]

        _ = self.get_image_features(images2, norm=True)
        attn_maps_B = self.model.vision_model.encoder.layers[target_layer].stores[0]

        return F.cosine_similarity(attn_maps_A.reshape(-1).unsqueeze(0), attn_maps_B.reshape(-1).unsqueeze(0))


