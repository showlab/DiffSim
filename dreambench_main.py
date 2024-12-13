import os
import csv
import argparse
import random
import json

from diffusers.utils import load_image
import torch
import torch.nn.functional as F
import numpy as np

import lpips
from PIL import Image

from diffsim.diffsim import DiffSim, process_image
from diffsim.diffsim_xl import diffsim_xl
from diffsim.diffsim_dit import diffsim_DiT
from metrics.clip_i import CLIPScore
from metrics.dino import Dinov2Score, DinoScore
from metrics.foreground_feature_averaging import ForegroundFeatureAveraging
from argprocess import arg_parse

IMAGE_EXT_LOWER = ["png", "jpeg", "jpg"]
IMAGE_EXT = IMAGE_EXT_LOWER + [_ext.upper() for _ext in IMAGE_EXT_LOWER]


def evaluate_similarity(args, image_path, device):
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

    # load the annotation file
    rating_path = os.path.join(image_path, "data_human_rating")

    with torch.no_grad():
        total_samples = 0
        correct_predictions = 0
        print(f"=========seed {args.seed}=========")
        print(f"Experiment on {args.target_block}, layer {args.target_layer}, timestep {args.target_step}:")
        for pipe_dir in os.listdir(image_path):
            # print("Image dir:", pipe_dir)
            if "blip_diffusion" in pipe_dir:
                json_name = "blip_diffusion-cp.json"
            elif "dreambooth" in pipe_dir:
                json_name = "dreambooth_sd-cp.json"
            elif "ip_adapter_plus_sdxl" in pipe_dir:
                json_name = "ip_adapter_plus_vit_h_sdxl-cp.json"
            elif "ip_adapter_sdxl" in pipe_dir:
                json_name = "ip_adapter_vit_g_sdxl-cp.json"
            elif "textual_inversion" in pipe_dir:
                json_name = "textual_inversion_sd-cp.json"
            else:
                continue
            
            with open(os.path.join(rating_path, "merged_data/group1/", json_name), 'r') as file:
                anno_1 = json.load(file)
            with open(os.path.join(rating_path, "merged_data/group2/", json_name), 'r') as file:
                anno_2 = json.load(file)
            
            pipe_image_path = os.path.join(image_path, pipe_dir)

            # load images
            image1_dir = os.path.join(pipe_image_path, "src_image")
            image2_dir = os.path.join(pipe_image_path, "tgt_image")
            text_dir = os.path.join(pipe_image_path, "text")

            for ref_image in os.listdir(image1_dir): # reference image, A (only the directory name)
                # print("Ref image:", ref_image)
                # averating the annotations from the two groups
                filtered_1 = {k: v for k, v in anno_1.items() if k.startswith(ref_image)}
                filtered_2 = {k: v for k, v in anno_2.items() if k.startswith(ref_image)}
                
                # Combine keys from annotation filtered 1 and 2
                result = {}
                for key, value in filtered_1.items():
                    if abs(value - filtered_2[key]) > 2: # abolish samples whose annotations from two groups diverse too much
                        continue
                    result[key] = (value + filtered_2[key]) / 2 # average the two annotations
                
                # select image pairs
                selected_pairs = {}
                for key_a, value_a in result.items():
                    for key_b, value_b in result.items():
                        if key_a == key_b or abs(value_a - value_b) < 2: # only if human score diverses over 1
                            continue
                        if (key_b, key_a) in selected_pairs:
                            continue
                        combined_key = (key_a, key_b)
                        selected_pairs[combined_key] = (0 if value_a > value_b else 1) # 0: image #key_a is better; 1: image #key_b is better
                
                selected_pairs = list(selected_pairs.items())
                if len(selected_pairs) > 5:
                    selected_pairs = random.sample(selected_pairs, 5)

                ref_image_file = os.path.join(image1_dir, ref_image, "0_0.jpg")
                for pair in selected_pairs:
                    tgt_image1_file = os.path.join(image2_dir, ref_image, f"{pair[0][0][-1]}_0.jpg")
                    tgt_image1_text = os.path.join(text_dir, ref_image, f"{pair[0][0][-1]}_0.txt")
                    tgt_image2_file = os.path.join(image2_dir, ref_image, f"{pair[0][1][-1]}_0.jpg")
                    tgt_image2_text = os.path.join(text_dir, ref_image, f"{pair[0][1][-1]}_0.txt")

                    text1 = open(tgt_image1_text, "r")
                    prompt1 = text1.readline().strip('\n')
                    text2 = open(tgt_image2_text, "r")
                    prompt2 = text2.readline().strip('\n')

                    if args.metric == 'diffsim':
                        diff_ab = diffsim.diffsim(image_A=ref_image_file,
                                                    image_B=tgt_image1_file,
                                                    img_size=args.image_size,
                                                    prompt=prompt,
                                                    target_block=args.target_block,
                                                    target_layer=args.target_layer,
                                                    target_step=args.target_step,
                                                    ip_adapter=args.ip_adapter,
                                                    seed=args.seed,
                                                    device=device,
                                                    similarity=args.similarity)
                        diff_ac = diffsim.diffsim(image_A=ref_image_file,
                                                    image_B=tgt_image2_file,
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
                        diff_ab = diffsim_xl_score.diffsim_score(ref_image_file, tgt_image1_file, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                        diff_ac = diffsim_xl_score.diffsim_score(ref_image_file, tgt_image2_file, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                    elif args.metric == 'dit':
                        diff_ab = diffsim_dit.diffsim_score(ref_image_file, tgt_image1_file, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                        diff_ac = diffsim_dit.diffsim_score(ref_image_file, tgt_image2_file, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                    elif args.metric == 'clip_i':
                        diff_ab = clip_score.clipi_score(load_image(ref_image_file), load_image(tgt_image1_file))[0]
                        diff_ac = clip_score.clipi_score(load_image(ref_image_file), load_image(tgt_image2_file))[0]
                    elif args.metric == 'clip_cross':
                        diff_ab = clip_score.clip_cross_score(load_image(ref_image_file), load_image(tgt_image1_file), args.target_layer)
                        diff_ac = clip_score.clip_cross_score(load_image(ref_image_file), load_image(tgt_image2_file), args.target_layer)
                    elif args.metric == 'dino' or args.metric == 'dinov1':
                        diff_ab = dino_score.dino_score(load_image(ref_image_file), load_image(tgt_image1_file))[0]
                        diff_ac = dino_score.dino_score(load_image(ref_image_file), load_image(tgt_image2_file))[0]
                    elif args.metric == 'dino_cross':
                        diff_ab = dino_score.dino_cross_score(load_image(ref_image_file), load_image(tgt_image1_file), args.target_layer)
                        diff_ac = dino_score.dino_cross_score(load_image(ref_image_file), load_image(tgt_image2_file), args.target_layer)
                    elif args.metric == 'cute':
                        diff_ab = cute_score("Crop-Feat", [load_image(ref_image_file)], [load_image(tgt_image1_file)])
                        diff_ac = cute_score("Crop-Feat", [load_image(ref_image_file)], [load_image(tgt_image2_file)])
                    elif args.metric == 'lpips':
                        diff_ab = lpips_score(process_image(load_image(ref_image_file)), process_image(load_image(tgt_image1_file))).item()
                        diff_ac = lpips_score(process_image(load_image(ref_image_file)), process_image(load_image(tgt_image2_file))).item()
                    elif args.metric == 'ensemble':
                        diff_ab = diffsim.diffsim(image_A=ref_image_file,
                                                image_B=tgt_image1_file,
                                                img_size=args.image_size,
                                                prompt=prompt,
                                                target_block=args.target_block,
                                                target_layer=args.target_layer,
                                                target_step=args.target_step,
                                                ip_adapter=args.ip_adapter,
                                                seed=args.seed,
                                                device=device,
                                                similarity=args.similarity)
                        diff_ac = diffsim.diffsim(image_A=ref_image_file,
                                                image_B=tgt_image2_file,
                                                img_size=args.image_size,
                                                prompt=prompt,
                                                target_block=args.target_block,
                                                target_layer=args.target_layer,
                                                target_step=args.target_step,
                                                ip_adapter=args.ip_adapter,
                                                seed=args.seed,
                                                device=device,
                                                similarity=args.similarity)
                        clip_ab = clip_score.clipi_score(load_image(ref_image_file), load_image(tgt_image1_file))[0]
                        clip_ac = clip_score.clipi_score(load_image(ref_image_file), load_image(tgt_image2_file))[0]
                        dino_ab = dino_score.dino_score(load_image(ref_image_file), load_image(tgt_image1_file))[0]
                        dino_ac = dino_score.dino_score(load_image(ref_image_file), load_image(tgt_image2_file))[0]


                    if args.metric == 'ensemble':
                        diff_corr = 0 if diff_ab < diff_ac else 1
                        clip_corr = 0 if clip_ab < clip_ac else 1
                        dino_corr = 0 if dino_ab < dino_ac else 1
                        if (pair[1] == 0 and diff_corr + clip_corr + dino_corr >= 2) or (pair[1] == 1 and diff_corr + clip_corr + dino_corr <= 1):
                            correct_predictions += 1
                    else:
                        compare_result = 0 if diff_ab > diff_ac else 1

                        if compare_result == pair[1]:
                            correct_predictions += 1

                    total_samples += 1

        print(f"Total samples now: {total_samples}")
        print(f'Current accuracy: {correct_predictions / total_samples * 100}%')    



if __name__ == "__main__":
    args = arg_parse()
    device = 'cuda'

    evaluate_similarity(args, args.image_path, device)
