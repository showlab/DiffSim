import os
import re
import argparse
import random

from diffusers.utils import load_image
import torch
import torch.nn.functional as F

import lpips
from PIL import Image

from diffsim.diffsim import DiffSim, process_image
from diffsim.diffsim_xl import diffsim_xl
from diffsim.diffsim_dit import diffsim_DiT
from metrics.clip_i import CLIPScore
from metrics.dino import Dinov2Score, DinoScore
from metrics.foreground_feature_averaging import ForegroundFeatureAveraging
from metrics.vgg_gram import vgg_gram
from argprocess import arg_parse



if __name__ == "__main__":
    args = arg_parse()
    device = 'cuda'

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
        total = 0
        correct = 0 # the similarity value between A & B is smaller than A & C
        correct_2x = 0 # the similarity value between A & B is twice smaller than A & C
        random.seed(args.seed)
        print(f"=========seed {args.seed}=========")
        print(f"Experiment on {args.target_block}, layer {args.target_layer}, timestep {args.target_step}:")
        for cls in os.listdir(args.image_path):
            if cls == "main.py" or cls == ".DS_Store":
                continue
            cls_dir_path = os.path.join(args.image_path, cls)
            # print("=============")
            # print(f"Traversing the image path {cls_dir_path}")
            # print("=============")
            for experiment in range(10):
                # print(f"Running experiment {experiment + 1}/10")

                for subdir_lvl1, dirs_lvl2, _ in os.walk(cls_dir_path):
                    for dir_lvl2 in dirs_lvl2:
                        dir_lvl2_path = os.path.join(subdir_lvl1, dir_lvl2)

                        subdirs_lvl3 = [d for d in os.listdir(dir_lvl2_path) if os.path.isdir(os.path.join(dir_lvl2_path, d))]

                        if not subdirs_lvl3:
                            continue

                        selected_lvl3_dir = random.choice(subdirs_lvl3)
                        selected_lvl3_dir_path = os.path.join(dir_lvl2_path, selected_lvl3_dir)

                        image_files = [f for f in os.listdir(selected_lvl3_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                        if len(image_files) < 2:
                            continue

                        image_A, image_B = random.sample(image_files, 2)

                        image_A_path = os.path.join(selected_lvl3_dir_path, image_A)
                        # print("Image A:", image_A_path)

                        image_B_path = os.path.join(selected_lvl3_dir_path, image_B)
                        # print("Image B:", image_B_path)

                        other_dirs_lvl2 = [d for d in dirs_lvl2 if d != dir_lvl2]

                        if not other_dirs_lvl2:
                            continue

                        selected_other_lvl2 = random.choice(other_dirs_lvl2)
                        selected_other_lvl2_path = os.path.join(subdir_lvl1, selected_other_lvl2)

                        selected_other_lvl3_dir_path = os.path.join(selected_other_lvl2_path, selected_lvl3_dir)

                        other_image_files = [f for f in os.listdir(selected_other_lvl3_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                        if not other_image_files:
                            continue

                        image_C = random.choice(other_image_files)
                        image_C_path = os.path.join(selected_other_lvl3_dir_path, image_C)
                        # print("Image C:", image_C_path)

                        prompt = f"The photo of a {cls}"

                        if args.metric == 'diffsim':
                            diff_ab = diffsim.diffsim(image_A=image_A_path,
                                                        image_B=image_B_path,
                                                        img_size=args.image_size,
                                                        prompt=prompt,
                                                        target_block=args.target_block,
                                                        target_layer=args.target_layer,
                                                        target_step=args.target_step,
                                                        ip_adapter=args.ip_adapter,
                                                        seed=args.seed,
                                                        device=device,
                                                        similarity=args.similarity)
                            diff_ac = diffsim.diffsim(image_A=image_A_path,
                                                        image_B=image_C_path,
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
                            diff_ab = diffsim_xl_score.diffsim_score(image_A_path, image_B_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                            diff_ac = diffsim_xl_score.diffsim_score(image_A_path, image_C_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                        elif args.metric == 'dit':
                            diff_ab = diffsim_dit.diffsim_score(image_A_path, image_B_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                            diff_ac = diffsim_dit.diffsim_score(image_A_path, image_C_path, args.image_size, prompt, args.target_block, args.target_layer, args.target_step, args.similarity, args.seed)
                        elif args.metric == 'clip_i':
                            diff_ab = clip_score.clipi_score(load_image(image_A_path), load_image(image_B_path))[0]
                            diff_ac = clip_score.clipi_score(load_image(image_A_path), load_image(image_C_path))[0]
                        elif args.metric == 'clip_cross':
                            diff_ab = clip_score.clip_cross_score(load_image(image_A_path), load_image(image_B_path), args.target_layer)
                            diff_ac = clip_score.clip_cross_score(load_image(image_A_path), load_image(image_C_path), args.target_layer)
                        elif args.metric == 'dino':
                            diff_ab = dino_score.dino_score(load_image(image_A_path), load_image(image_B_path))[0]
                            diff_ac = dino_score.dino_score(load_image(image_A_path), load_image(image_C_path))[0]
                        elif args.metric == 'dino_cross':
                            diff_ab = dino_score.dino_cross_score(load_image(image_A_path), load_image(image_B_path), args.target_layer)
                            diff_ac = dino_score.dino_cross_score(load_image(image_A_path), load_image(image_C_path), args.target_layer)
                        elif args.metric == 'cute':
                            diff_ab = cute_score("Crop-Feat", [load_image(image_A_path)], [load_image(image_B_path)])
                            diff_ac = cute_score("Crop-Feat", [load_image(image_A_path)], [load_image(image_C_path)])
                        elif args.metric == 'lpips':
                            diff_ab = lpips_score(process_image(load_image(image_A_path)), process_image(load_image(image_B_path)))
                            diff_ac = lpips_score(process_image(load_image(image_A_path)), process_image(load_image(image_C_path)))
                        # elif args.metric == 'gram':
                        #     diff_ab = gram_score(image_A_path, image_B_path)
                        #     diff_ac = gram_score(image_A_path, image_C_path)
                        elif args.metric == 'ensemble':
                            diff_ab = diffsim(image_A=image_A_path,
                                        image_B=image_B_path,
                                        img_size=args.image_size,
                                        prompt=prompt,
                                        target_block=args.target_block,
                                        target_layer=args.target_layer,
                                        target_step=args.target_step,
                                        ip_adapter=args.ip_adapter,
                                        seed=args.seed,
                                        device=device,
                                        similarity=args.similarity)
                            diff_ac = diffsim(image_A=image_A_path,
                                        image_B=image_C_path,
                                        img_size=args.image_size,
                                        prompt=prompt,
                                        target_block=args.target_block,
                                        target_layer=args.target_layer,
                                        target_step=args.target_step,
                                        ip_adapter=args.ip_adapter,
                                        seed=args.seed,
                                        device=device,
                                        similarity=args.similarity)
                            clip_ab = clip_score.clipi_score(load_image(image_A_path), load_image(image_B_path))[0]
                            clip_ac = clip_score.clipi_score(load_image(image_A_path), load_image(image_C_path))[0]
                            dino_ab = dino_score.dino_score(load_image(image_A_path), load_image(image_B_path))[0]
                            dino_ac = dino_score.dino_score(load_image(image_A_path), load_image(image_C_path))[0]


                        if args.metric == 'ensemble':
                            diff_corr = 0 if diff_ab < diff_ac else 1
                            clip_corr = 0 if clip_ab < clip_ac else 1
                            dino_corr = 0 if dino_ab < dino_ac else 1
                            if diff_corr + clip_corr + dino_corr >= 2:
                                correct += 1
                        else:
                            if args.similarity == 'mse' or args.metric == 'lpips':
                                if diff_ab < diff_ac:
                                    correct += 1
                                if diff_ab * 2 < diff_ac:
                                    correct_2x += 1
                            elif args.similarity == 'cosine':
                                if diff_ab > diff_ac:
                                    correct += 1
                                if diff_ab > 2 * diff_ac:
                                    correct_2x += 1
        
                        total += 1
                        if total % 450 == 0:
                            print(f"Current total samples: {total}")
                            if total > 0:
                                # accuracy = correct / total * 100
                                # accuracy_2x = correct_2x / total * 100
                                # print(f"Accuracy: {accuracy}%")
                                # print("Image C:", image_C_path)
                                # # print(f"2x Accuracy: {accuracy_2x}%")
                                print(f"Total {total}; Correct {correct}; Correct 2x {correct_2x}")
                                print(f"Accuracy: {correct / total * 100}%")
                                print(f"2x Accuracy: {correct_2x / total * 100}%")
                            else:
                                print("No valid comparisons were made.")
    
    print(f"Total comparisons: {total}")
    if total > 0:
        print(f"Total {total}; Correct {correct}; Correct 2x {correct_2x}")
        print(f"Accuracy: {correct / total * 100}%")
        print(f"2x Accuracy: {correct_2x / total * 100}%")
    else:
        print("No valid comparisons were made.")

    