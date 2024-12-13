export CUDA_VISIBLE_DEVICES=1

# Sref
# python -u style_main.py --image_path /tiamat-NAS/songyiren/dataset/Sref508/ --target_block "up_blocks" --target_layer 0 --target_step 900 --similarity "cosine" --seed 2334 --metric "diffsim"

# InstantStyle
python -u style_main.py --image_path /tiamat-NAS/songyiren/projects/SDmetric/DiffSim0/stylesim-test --target_block "up_blocks" --target_layer 0 --target_step 900 --similarity "cosine" --seed 2334 --metric "diffsim"
