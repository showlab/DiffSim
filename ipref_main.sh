export CUDA_VISIBLE_DEVICES=2

# ipref1
python -u ipref_main.py --image_path /tiamat-NAS/songyiren/dataset/IPref --original_path "/tiamat-NAS/songyiren/dataset/IPA-bench" --target_block "up_blocks" --target_layer 5 --target_step 750 --similarity "cosine" --seed 2334 --metric "diffsim"
