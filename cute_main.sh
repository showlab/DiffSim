export CUDA_VISIBLE_DEVICES=2

python -u cute_main.py --image_path "/tiamat-NAS/songyiren/dataset/CUTE/" --image_size 512 --target_block "up_blocks" --target_layer 0 --target_step 600 --similarity "cosine" --seed 2334 --metric "diffsim"

