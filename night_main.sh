export CUDA_VISIBLE_DEVICES=1

python -u night_main.py --image_path /tiamat-NAS/songyiren/projects/SDmetric/DreamSim/dreamsim/dataset/nights/ --image_size 512 --target_block "up_blocks" --target_layer 0 --target_step 500 --similarity "cosine" --seed 2334 --metric "diffsim"
