#!/bin/bash

# evaluate validation subset

torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 29438 \
/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/evaluate.py \
--rcd_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/test_validation_subset' \
--rcd_file 'test_validation_subset' \
--visualization False \
--deep_supervision False \
--text_prompts_json '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/challenge_data/CVPR25_TextSegFMData_with_class.json' \
--datasets_jsonl '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/challenge_data/validation_subset.jsonl' \
--online_crop True \
--crop_size 288 288 96 \
--vision_backbone 'UNET' \
--checkpoint '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/checkpoints/nano_cvpr25_v0.pth' \
--partial_load True \
--text_encoder 'ours' \
--text_encoder_checkpoint '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/checkpoints/text_encoder_cvpr25_v0.pth' \
--batchsize_3d 2 \
--max_queries 256 \
--pin_memory False \
--num_workers 8 \
--dice True \
--nsd True