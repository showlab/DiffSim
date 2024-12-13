import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Parse some command-line arguments.")
    parser.add_argument('--image_path', type=str, help='Path to image folder')
    parser.add_argument('--original_path', type=str, default=None, help='Path to original images for ipref')
    parser.add_argument('--out_path', type=str, help='Path to the output folder (can be ckpt folder / retrieval result folder)')
    parser.add_argument('--image_size', type=int, default=512, help="(Resized) Resolution of compared image")
    parser.add_argument('--target_block', type=str, choices=['down_blocks', 'mid_blocks', 'up_blocks'], default='up_blocks', help='Where the target layer lies within')
    parser.add_argument('--target_layer', type=int, default=2, nargs='+', help='The No. of target layer to calculate the metric. For SD XL, please specify 3 numbers to indicate block_id, trans_id and attention_id')
    parser.add_argument('--target_step', type=int, default=100, help='The target denoising timestep to calculate the metric')
    parser.add_argument('--metric', type=str, choices=['diffsim', 'diffsim_xl', 'clip_i', 'clip_cross', 'dino', 'dinov1', 'dino_cross', 'cute', 'lpips', 'gram', 'diffeats', 'clipfeats', 'dinofeats', 'ensemble', 'dit'], default='diffsim')
    parser.add_argument('--similarity', type=str, choices=['cosine', 'mse'], default='mse', help='How to calculate the similary between attention maps')
    parser.add_argument('--prompt', type=str, default='High quality image', help='Prompt used to specify the target region in original image')
    parser.add_argument('--ip_adapter', action='store_true', help='Whether use IP-Adapter Plus\'s image cross attention layer instead of pure self attention layer')
    parser.add_argument('--use_mask', action='store_true', help='Whether use SAM-CLIP to segment the target region first')
    parser.add_argument('--use_text_attn', action='store_true', help='Whether use the cross-attention results of text to guide conditional similarity')
    parser.add_argument('--seed', type=int, default=2333, help='Seed')

    return parser.parse_args()