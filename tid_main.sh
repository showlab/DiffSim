export CUDA_VISIBLE_DEVICES=5

python -u tid_main.py --image_path /tiamat-NAS/songyiren/dataset/tid2013 --target_block "up_blocks" --target_layer 0 --target_step 900 --similarity "cosine" --seed 2334 --metric "diffsim"
