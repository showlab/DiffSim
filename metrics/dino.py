import math
import os
from typing import Literal, TypeAlias

import fire
import megfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import BitImageProcessor, Dinov2Model

import PIL
import numpy as np

from metrics.hooks import dinov2_self_attention_forward_hook, dinov2_self_attention_forward_feature_hook

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
_DEFAULT_MODEL_V1: str = "dino_vits8"
_DEFAULT_MODEL_V2: str = MODEL_ZOOS["facebook/dinov2-small"]
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32

ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor

class DinoScore:
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V1,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = torch.hub.load("facebookresearch/dino:main", model_or_name_path).to(device, dtype=torch_dtype)
        self.model.eval()
        self.processor = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = [self.processor(i) for i in image]
        inputs = torch.stack(inputs).to(self.device, dtype=self.dtype)
        image_features = self.model(inputs)
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def dino_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
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


class Dinov2Score(DinoScore):
    # NOTE: noqa, in version 1, the performance of the official repository and HuggingFace is inconsistent.
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V2,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = Dinov2Model.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = BitImageProcessor.from_pretrained(model_or_name_path, local_files_only=local_files_only)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model(inputs["pixel_values"].to(self.device, dtype=self.dtype)).last_hidden_state[:, 0, :]
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features
    
    def attention_calc(self, q, k, v, attention_head_size, dropout):
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        return context_layer

    @torch.no_grad()
    def dino_cross_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType], target_layer: int):
        if len(target_layer) == 1:
            target_layer = target_layer[0]
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        # print(f"DinoV2Model has {len(self.model.encoder.layer)} layers") --> 12 layers in total
        self.model.encoder.layer[target_layer].attention.attention.register_forward_hook(dinov2_self_attention_forward_hook)
        attention_head_size, dropout = self.model.encoder.layer[target_layer].attention.attention.attention_head_size, self.model.encoder.layer[target_layer].attention.attention.dropout

        _ = self.get_image_features(images1, norm=True)
        queryA, keyA, valueA = self.model.encoder.layer[target_layer].attention.attention.stores

        _ = self.get_image_features(images2, norm=True)
        queryB, keyB, valueB = self.model.encoder.layer[target_layer].attention.attention.stores

        attn_a_on_b = self.attention_calc(queryA, keyB, valueB, attention_head_size, dropout)
        attn_b_on_a = self.attention_calc(queryB, keyA, valueA, attention_head_size, dropout)
        self_attn_a = self.attention_calc(queryA, keyA, valueA, attention_head_size, dropout)
        self_attn_b = self.attention_calc(queryB, keyB, valueB, attention_head_size, dropout)

        diffsim_a_on_b = F.cosine_similarity(attn_a_on_b.reshape(-1).unsqueeze(0), self_attn_a.reshape(-1).unsqueeze(0))
        diffsim_b_on_a = F.cosine_similarity(attn_b_on_a.reshape(-1).unsqueeze(0), self_attn_b.reshape(-1).unsqueeze(0))

        return (diffsim_a_on_b + diffsim_b_on_a) / 2
    
    @torch.no_grad()
    def dino_feature_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType], target_layer: int):
        if len(target_layer) == 1:
            target_layer = target_layer[0]
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        # print(f"DinoV2Model has {len(self.model.encoder.layer)} layers") --> 12 layers in total
        self.model.encoder.layer[target_layer].attention.attention.register_forward_hook(dinov2_self_attention_forward_feature_hook)
        attention_head_size, dropout = self.model.encoder.layer[target_layer].attention.attention.attention_head_size, self.model.encoder.layer[target_layer].attention.attention.dropout

        _ = self.get_image_features(images1, norm=True)
        attn_maps_A = self.model.encoder.layer[target_layer].attention.attention.stores[0]

        _ = self.get_image_features(images2, norm=True)
        attn_maps_B = self.model.encoder.layer[target_layer].attention.attention.stores[0]

        return F.cosine_similarity(attn_maps_A.reshape(-1).unsqueeze(0), attn_maps_B.reshape(-1).unsqueeze(0))
