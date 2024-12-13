export CUDA_VISIBLE_DEVICES=1

python -u dreambench_main.py --image_path /tiamat-NAS/songyiren/Xiaokang/dreambench_plus/samples/ --target_block "up_blocks" --target_layer 0 --target_step 750 --similarity "cosine" --seed 2334 --metric "diffsim"
