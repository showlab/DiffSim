import os
import re
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt

from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, CrossAttnDownBlock2D
from diffusers.utils import PIL_INTERPOLATION
import torch
import torch.nn.functional as F
import numpy as np

from segment_anything import build_sam, SamAutomaticMaskGenerator
import lpips

from diffsim.hacked_modules import hacked_CrossAttnUpBlock2D_forward
from diffsim.diffsim_pipeline import DiffSimPipeline
from diffsim.diffsim import diffsim, diffsim_value, process_image
from diffsim.diffsim_xl import diffsim_xl
from metrics.clip_i import CLIPScore
from metrics.dino import Dinov2Score
from metrics.foreground_feature_averaging import ForegroundFeatureAveraging
from metrics.vgg_gram import vgg_gram
from argprocess import arg_parse


def sref_vis():
    # A, B, C from the same background
    # args.target_layer[2] = layer
    total = 0
    correct = 0 # the similarity value between A & B is smaller than A & C
    correct_2x = 0 # the similarity value between A & B is twice smaller than A & C
    origin_path = "/tiamat-NAS/songyiren/dataset/Sref508/"
    diffsim_path = "/tiamat-NAS/songyiren/Xiaokang/data/diffsim_sref_ckpt/l0_500_retrieval/"
    clip_path = "/tiamat-NAS/songyiren/Xiaokang/data/clip_sref_retrieval/"
    dino_path = "/tiamat-NAS/songyiren/Xiaokang/data/dino_sref_retrieval/"
    out_path = "/tiamat-NAS/songyiren/Xiaokang/data/sref_retrieval_comparison/"
    prompt = "A high quality image"
    for cls in os.listdir(diffsim_path):
        if cls == "main.py" or cls == ".DS_Store":
            continue
        cls_dir_path = os.path.join(diffsim_path, cls)
        cls_ckpt_path = os.path.join(out_path, cls)
        if not os.path.exists(cls_ckpt_path):
            print(f"Make dir {cls_ckpt_path}")
            os.makedirs(cls_ckpt_path)
        for retrieval_result in os.listdir(cls_dir_path):
            retrieval_result_clip = os.path.join(clip_path, cls, retrieval_result)
            retrieval_result_dino = os.path.join(dino_path, cls, retrieval_result)
            retrieval_result_diffsim = os.path.join(diffsim_path, cls, retrieval_result)

            txt_files = [retrieval_result_diffsim, retrieval_result_clip, retrieval_result_dino]
            
            def read_image_path(dirs, file):
                image_paths = []
                with open(file, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls, img_id = parts[0].split('_')
                            image_path = os.path.join(dirs, cls, f"{img_id[:-1]}.png")
                            image_paths.append(image_path)
                            if len(image_paths) >= 4:  # Read only the first 4 image paths
                                break
                return image_paths
            
            origin_image_path = os.path.join(origin_path, cls, retrieval_result.replace('txt', 'png'))
            image_paths_grid = []
            image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_diffsim))
            image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_clip))
            image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_dino))
            # print(image_paths_grid)
            
            # Display the images in a 3x5 grid and save the output
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))

            for row, row_images in enumerate(image_paths_grid):
                for col, img_path in enumerate(row_images):
                    # print(img_path)
                    img = Image.open(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')  # Turn off axis for better display

            # Set padding between images
            plt.subplots_adjust(wspace=0.2, hspace=0.2)

            # Save the grid as an image
            plt.tight_layout()
            
            save_path = os.path.join(cls_ckpt_path, retrieval_result.replace('txt', 'png'))
            plt.savefig(save_path)
            print(f"Result save to {save_path}")
            plt.close()
    
    print("Finish")


def coco_vis():
    # A, B, C from the same background
    # args.target_layer[2] = layer
    total = 0
    correct = 0 # the similarity value between A & B is smaller than A & C
    correct_2x = 0 # the similarity value between A & B is twice smaller than A & C
    origin_path = "/tiamat-NAS/data/coco/test2017"
    diffsim_path = "/tiamat-NAS/songyiren/Xiaokang/data/diffsim_coco_ckpt/l0_500_test_retrieval/"
    clip_path = "/tiamat-NAS/songyiren/Xiaokang/data/clip_coco_retrieval/"
    dino_path = "/tiamat-NAS/songyiren/Xiaokang/data/dino_coco_retrieval/"
    out_path = "/tiamat-NAS/songyiren/Xiaokang/data/coco_retrieval_comparison/"
    prompt = "A high quality image"

    for retrieval_result in os.listdir(diffsim_path):
        retrieval_result_clip = os.path.join(clip_path, retrieval_result)
        retrieval_result_dino = os.path.join(dino_path, retrieval_result)
        retrieval_result_diffsim = os.path.join(diffsim_path, retrieval_result)

        txt_files = [retrieval_result_diffsim, retrieval_result_clip, retrieval_result_dino]
        
        def read_image_path(dirs, file):
            image_paths = []
            with open(file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        img_id = parts[0]
                        image_path = os.path.join(dirs, f"{img_id[:-1]}.jpg")
                        image_paths.append(image_path)
                        if len(image_paths) >= 4:  # Read only the first 4 image paths
                            break
            return image_paths
        
        origin_image_path = os.path.join(origin_path, retrieval_result.replace('txt', 'jpg'))
        image_paths_grid = []
        image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_diffsim))
        image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_clip))
        image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_dino))
        # print(image_paths_grid)
        
        # Display the images in a 3x5 grid and save the output
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))

        for row, row_images in enumerate(image_paths_grid):
            for col, img_path in enumerate(row_images):
                # print(img_path)
                img = Image.open(img_path)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')  # Turn off axis for better display

        # Set padding between images
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # Save the grid as an image
        plt.tight_layout()
        
        save_path = os.path.join(out_path, retrieval_result.replace('txt', 'png'))
        plt.savefig(save_path)
        print(f"Result save to {save_path}")
        plt.close()

def ip_vis():
    # A, B, C from the same background
    # args.target_layer[2] = layer
    total = 0
    correct = 0 # the similarity value between A & B is smaller than A & C
    correct_2x = 0 # the similarity value between A & B is twice smaller than A & C
    origin_path = "/tiamat-NAS/songyiren/dataset/ipref_combine/"
    diffsim_path = "/tiamat-NAS/songyiren/Xiaokang/data/ipref_combine_retrieval/diffsim_retrieval"
    clip_path = "/tiamat-NAS/songyiren/Xiaokang/data/ipref_combine_retrieval/clip_retrieval"
    dino_path = "/tiamat-NAS/songyiren/Xiaokang/data/ipref_combine_retrieval/dino_retrieval"
    out_path = "/tiamat-NAS/songyiren/Xiaokang/data/ipref_combine_retrieval/retrieval_comparison"
    prompt = "A high quality image"

    for cls in os.listdir(diffsim_path):
        if cls == "main.py" or cls == ".DS_Store":
            continue
        cls_dir_path = os.path.join(diffsim_path, cls)
        cls_ckpt_path = os.path.join(out_path, cls)
        if not os.path.exists(cls_ckpt_path):
            print(f"Make dir {cls_ckpt_path}")
            os.makedirs(cls_ckpt_path)
        for retrieval_result in os.listdir(cls_dir_path):
            retrieval_result_clip = os.path.join(clip_path, cls, retrieval_result)
            retrieval_result_dino = os.path.join(dino_path, cls, retrieval_result)
            retrieval_result_diffsim = os.path.join(diffsim_path, cls, retrieval_result)

            txt_files = [retrieval_result_diffsim, retrieval_result_clip, retrieval_result_dino]
            
            def read_image_path(dirs, file):
                image_paths = []
                with open(file, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls, img_id = parts[0].split('_')
                            if img_id == "1:":
                                continue
                            # print(line, cls, img_id)
                            image_path = os.path.join(dirs, cls, f"{img_id[:-1]}.png")
                            image_paths.append(image_path)
                            if len(image_paths) >= 4:  # Read only the first 4 image paths
                                break
                return image_paths
            
            origin_image_path = os.path.join(origin_path, cls, retrieval_result.replace('txt', 'png'))
            image_paths_grid = []
            image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_diffsim))
            image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_clip))
            image_paths_grid.append([origin_image_path] + read_image_path(origin_path, retrieval_result_dino))
            # print(image_paths_grid)
            
            # Display the images in a 3x5 grid and save the output
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))

            for row, row_images in enumerate(image_paths_grid):
                for col, img_path in enumerate(row_images):
                    img = Image.open(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')  # Turn off axis for better display

            # Set padding between images
            plt.subplots_adjust(wspace=0.2, hspace=0.2)

            # Save the grid as an image
            plt.tight_layout()
            
            save_path = os.path.join(cls_ckpt_path, retrieval_result.replace('txt', 'png'))
            plt.savefig(save_path)
            print(f"Result save to {save_path}")
            plt.close()
    
    print("Finish")


if __name__ == "__main__":
    # save_image_descriptors()
    sref_vis()
    # coco_vis()
    # ip_vis()
