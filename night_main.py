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


def night_eval():
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
    

    with torch.no_grad():
        print(f"=========seed {args.seed}=========")
        print(f"Experiment on {args.target_block}, layer {args.target_layer}, timestep {args.target_step}:")
        
        with open(os.path.join(args.image_path, "data.csv"), mode='r') as file:
            reader = csv.DictReader(file)

            total_samples = 0
            correct_predictions = 0
            for row in reader:
                if row['split'] != 'val':
                    continue

                ref_path = os.path.join(args.image_path, row['ref_path'])
                left_path = os.path.join(args.image_path, row['left_path'])
                right_path = os.path.join(args.image_path, row['right_path'])
                vote = int(row['left_vote'])

                prompt = f"An image of a {row['prompt'].lower()}"

                if args.metric == 'diffsim':
                    diff_ab = diffsim.diffsim(image_A=ref_path,
                                                image_B=left_path,
                                                img_size=args.image_size,
                                                prompt=prompt,
                                                target_block=args.target_block,
                                                target_layer=args.target_layer,
                                                target_step=args.target_step,
                                                ip_adapter=args.ip_adapter,
                                                seed=args.seed,
                                                device=device,
                                                similarity=args.similarity)
                    diff_ac = diffsim.diffsim(image_A=ref_path,
                                                image_B=right_path,
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
                    diff_ab = diffsim_xl_score.diffsim_score(ref_path, left_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                    diff_ac = diffsim_xl_score.diffsim_score(ref_path, right_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                elif args.metric == 'dit':
                    diff_ab = diffsim_dit.diffsim_score(ref_path, left_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                    diff_ac = diffsim_dit.diffsim_score(ref_path, right_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                elif args.metric == 'clip_i':
                    diff_ab = clip_score.clipi_score(load_image(ref_path), load_image(left_path))[0]
                    diff_ac = clip_score.clipi_score(load_image(ref_path), load_image(right_path))[0]
                elif args.metric == 'clip_cross':
                    diff_ab = clip_score.clip_cross_score(load_image(ref_path), load_image(left_path), args.target_layer)
                    diff_ac = clip_score.clip_cross_score(load_image(ref_path), load_image(right_path), args.target_layer)
                elif args.metric == 'clipfeats':
                    diff_ab = clip_score.clip_feature_score(load_image(ref_path), load_image(left_path), args.target_layer)
                    diff_ac = clip_score.clip_feature_score(load_image(ref_path), load_image(right_path), args.target_layer)
                elif args.metric == 'dino' or args.metric == 'dinov1':
                    diff_ab = dino_score.dino_score(load_image(ref_path), load_image(left_path))[0]
                    diff_ac = dino_score.dino_score(load_image(ref_path), load_image(right_path))[0]
                elif args.metric == 'dino_cross':
                    diff_ab = dino_score.dino_cross_score(load_image(ref_path), load_image(left_path), args.target_layer)
                    diff_ac = dino_score.dino_cross_score(load_image(ref_path), load_image(right_path), args.target_layer)
                elif args.metric == 'dinofeats':
                    diff_ab = dino_score.dino_feature_score(load_image(ref_path), load_image(left_path), args.target_layer)
                    diff_ac = dino_score.dino_feature_score(load_image(ref_path), load_image(right_path), args.target_layer)
                elif args.metric == 'cute':
                    diff_ab = cute_score("Crop-Feat", [load_image(ref_path)], [load_image(left_path)])
                    diff_ac = cute_score("Crop-Feat", [load_image(ref_path)], [load_image(right_path)])
                elif args.metric == 'lpips':
                    diff_ab = lpips_score(process_image(load_image(ref_path)), process_image(load_image(left_path))).item()
                    diff_ac = lpips_score(process_image(load_image(ref_path)), process_image(load_image(right_path))).item()
                elif args.metric == 'ensemble':
                    diff_ab = diffsim.diffsim(image_A=ref_path,
                                                image_B=left_path,
                                                img_size=args.image_size,
                                                prompt=prompt,
                                                target_block=args.target_block,
                                                target_layer=args.target_layer,
                                                target_step=args.target_step,
                                                ip_adapter=args.ip_adapter,
                                                seed=args.seed,
                                                device=device,
                                                similarity=args.similarity)
                    diff_ac = diffsim.diffsim(image_A=ref_path,
                                                image_B=right_path,
                                                img_size=args.image_size,
                                                prompt=prompt,
                                                target_block=args.target_block,
                                                target_layer=args.target_layer,
                                                target_step=args.target_step,
                                                ip_adapter=args.ip_adapter,
                                                seed=args.seed,
                                                device=device,
                                                similarity=args.similarity)
                    clip_ab = clip_score.clipi_score(load_image(ref_path), load_image(left_path))[0]
                    clip_ac = clip_score.clipi_score(load_image(ref_path), load_image(right_path))[0]
                    dino_ab = dino_score.dino_score(load_image(ref_path), load_image(left_path))[0]
                    dino_ac = dino_score.dino_score(load_image(ref_path), load_image(right_path))[0]

                if args.metric == 'ensemble':
                    diff_corr = 0 if diff_ab < diff_ac else 1
                    clip_corr = 0 if clip_ab < clip_ac else 1
                    dino_corr = 0 if dino_ab < dino_ac else 1
                    if (vote == 1 and diff_corr + clip_corr + dino_corr >= 2) or (vote == 0 and diff_corr + clip_corr + dino_corr <= 1):
                        correct_predictions += 1
                else:
                    if args.similarity == 'mse' or args.metric in ['lpips', 'dreamsim']:
                        predicted = 1 if diff_ab < diff_ac else 0
                    elif args.similarity == 'cosine':
                        predicted = 1 if diff_ab > diff_ac else 0

                    if predicted == vote:
                        correct_predictions += 1

                total_samples += 1

                if total_samples % 100 == 0:
                    accuracy = correct_predictions / total_samples * 100 if total_samples > 0 else 0
                    print(f"Total samples now: {total_samples}")
                    print(f'Validation accuracy: {accuracy:.2f}%')

            accuracy = correct_predictions / total_samples * 100 if total_samples > 0 else 0
            print(f'Final validation accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    night_eval()
