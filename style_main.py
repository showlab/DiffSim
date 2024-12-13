import os
import csv
import argparse
import random
import json

from diffusers.utils import load_image
import torch
import torch.nn.functional as F
import numpy as np

import megfile
import lpips

from diffsim.diffsim import DiffSim, process_image
from diffsim.diffsim_xl import diffsim_xl
from diffsim.diffsim_dit import diffsim_DiT
from metrics.clip_i import CLIPScore
from metrics.dino import Dinov2Score, DinoScore
from metrics.foreground_feature_averaging import ForegroundFeatureAveraging
from argprocess import arg_parse


def sref():
    args = arg_parse()
    device = 'cuda'
    random.seed(args.seed)

    # init sd model
    prompt = args.prompt
    if args.metric == 'diffsim' or args.metric == 'diffeats' or args.metric == 'ensemble':
        diffsim = DiffSim(torch.float16, device, args.ip_adapter)
    if args.metric == 'diffsim_xl':
        diffsim_xl_score = diffsim_xl(torch.float16, device, args.ip_adapter)        
    if args.metric == 'dit':
        diffsim_dit = diffsim_DiT(args.image_size, args.target_step, device)
    if 'clip' in args.metric or args.metric == 'ensemble':
        clip_score = CLIPScore(device=device)
    if args.metric == 'dino' or args.metric == 'dino_cross' or args.metric == 'dinofeats' or args.metric == 'ensemble':
        dino_score = Dinov2Score(device=device)
    if args.metric == 'dinov1':
        dino_score = DinoScore(device=device)
    if 'cute' in args.metric:
        cute_score = ForegroundFeatureAveraging(device=device)
    if 'lpips' in args.metric:
        lpips_score = lpips.LPIPS(net='vgg')

    subdir_dict = {}
    for root, dirs, _ in os.walk(args.image_path):
        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            images = [os.path.join(full_dir_path, f) for f in os.listdir(full_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) >= 2:
                subdir_dict[full_dir_path] = images

    subdir_paths = list(subdir_dict.keys())

    with torch.no_grad():
        print(f"=========seed {args.seed}=========")
        print(f"Experiment on {args.target_block}, layer {args.target_layer}, timestep {args.target_step}:")

        total = 0
        correct = 0
        correct_2x = 0
        for experiment in range(2000):
            if len(subdir_paths) < 2:
                continue
            
            # Select two different directories for A, B and C
            dir_A, dir_C = random.sample(subdir_paths, 2)
            
            # Select A and B from the same directory
            image_A, image_B = random.sample(subdir_dict[dir_A], 2)
            
            # Select C from a different directory
            image_C = random.choice(subdir_dict[dir_C])

            # Calculate similarity
            if args.metric == 'diffsim':
                diff_ab = diffsim.diffsim(image_A=image_A,
                                            image_B=image_B,
                                            img_size=args.image_size,
                                            prompt=prompt,
                                            target_block=args.target_block,
                                            target_layer=args.target_layer,
                                            target_step=args.target_step,
                                            ip_adapter=args.ip_adapter,
                                            seed=args.seed,
                                            device=device,
                                            similarity=args.similarity)
                diff_ac = diffsim.diffsim(image_A=image_A,
                                            image_B=image_C,
                                            img_size=args.image_size,
                                            prompt=prompt,
                                            target_block=args.target_block,
                                            target_layer=args.target_layer,
                                            target_step=args.target_step,
                                            ip_adapter=args.ip_adapter,
                                            seed=args.seed,
                                            device=device,
                                            similarity=args.similarity)
            elif args.metric == 'diffsim_xl':
                diff_ab = diffsim_xl_score.diffsim_score(image_A, image_B, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                diff_ac = diffsim_xl_score.diffsim_score(image_A, image_C, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
            elif args.metric == 'dit':
                diff_ab = diffsim_dit.diffsim_score(image_A, image_B, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                diff_ac = diffsim_dit.diffsim_score(image_A, image_C, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
            elif args.metric == 'clip_i':
                diff_ab = clip_score.clipi_score(load_image(image_A), load_image(image_B))[0]
                diff_ac = clip_score.clipi_score(load_image(image_A), load_image(image_C))[0]
            elif args.metric == 'clip_cross':
                diff_ab = clip_score.clip_cross_score(load_image(image_A), load_image(image_B), args.target_layer)
                diff_ac = clip_score.clip_cross_score(load_image(image_A), load_image(image_C), args.target_layer)
            elif args.metric == 'clipfeats':
                diff_ab = clip_score.clip_feature_score(load_image(image_A), load_image(image_B), args.target_layer)
                diff_ac = clip_score.clip_feature_score(load_image(image_A), load_image(image_C), args.target_layer)
            elif args.metric == 'dino' or args.metric == 'dinov1':
                diff_ab = dino_score.dino_score(load_image(image_A), load_image(image_B))[0]
                diff_ac = dino_score.dino_score(load_image(image_A), load_image(image_C))[0]
            elif args.metric == 'dino_cross':
                diff_ab = dino_score.dino_cross_score(load_image(image_A), load_image(image_B), args.target_layer)
                diff_ac = dino_score.dino_cross_score(load_image(image_A), load_image(image_C), args.target_layer)
            elif args.metric == 'dinofeats':
                diff_ab = dino_score.dino_feature_score(load_image(image_A), load_image(image_B), args.target_layer)
                diff_ac = dino_score.dino_feature_score(load_image(image_A), load_image(image_C), args.target_layer)
            elif args.metric == 'cute':
                diff_ab = cute_score("Crop-Feat", [load_image(image_A)], [load_image(image_B)])
                diff_ac = cute_score("Crop-Feat", [load_image(image_A)], [load_image(image_C)])
            elif args.metric == 'lpips':
                diff_ab = lpips_score(process_image(load_image(image_A)), process_image(load_image(image_B))).item()
                diff_ac = lpips_score(process_image(load_image(image_A)), process_image(load_image(image_C))).item()
            elif args.metric == 'ensemble':
                diff_ab = diffsim.diffsim(image_A=image_A,
                                            image_B=image_B,
                                            img_size=args.image_size,
                                            prompt=prompt,
                                            target_block=args.target_block,
                                            target_layer=args.target_layer,
                                            target_step=args.target_step,
                                            ip_adapter=args.ip_adapter,
                                            seed=args.seed,
                                            device=device,
                                            similarity=args.similarity)
                diff_ac = diffsim.diffsim(image_A=image_A,
                                            image_B=image_C,
                                            img_size=args.image_size,
                                            prompt=prompt,
                                            target_block=args.target_block,
                                            target_layer=args.target_layer,
                                            target_step=args.target_step,
                                            ip_adapter=args.ip_adapter,
                                            seed=args.seed,
                                            device=device,
                                            similarity=args.similarity)
                clip_ab = clip_score.clipi_score(load_image(image_A), load_image(image_B))[0]
                clip_ac = clip_score.clipi_score(load_image(image_A), load_image(image_C))[0]
                dino_ab = dino_score.dino_score(load_image(image_A), load_image(image_B))[0]
                dino_ac = dino_score.dino_score(load_image(image_A), load_image(image_C))[0]


            if args.metric == 'ensemble':
                diff_corr = 0 if diff_ab < diff_ac else 1
                clip_corr = 0 if clip_ab < clip_ac else 1
                dino_corr = 0 if dino_ab < dino_ac else 1
                if diff_corr + clip_corr + dino_corr >= 2:
                    correct += 1
            else:
                # Evaluate correctness based on similarity metric
                if args.similarity == 'mse' or args.metric in ['lpips', 'dreamsim']:
                    if diff_ab.item() < diff_ac.item():
                        correct += 1
                    if diff_ab.item() * 2 < diff_ac.item():
                        correct_2x += 1
                elif args.similarity == 'cosine':
                    if diff_ab > diff_ac:
                        correct += 1
                    if diff_ab > 2 * diff_ac:
                        correct_2x += 1
            
            total += 1

        # Output results
        if total > 0:
            accuracy = correct / total * 100
            accuracy_2x = correct_2x / total * 100
            print(f"Total comparisons: {total}")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"2x Accuracy: {accuracy_2x:.2f}%")
        else:
            print("No valid comparisons were made.")


if __name__ == '__main__':
    sref()
