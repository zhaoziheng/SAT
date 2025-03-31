#!/bin/bash

torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--master_port 29711 \
/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/train.py \
--log_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/log' \
--name 'nano_CVPR2025_w_pretrain(nnunet_aug_w_flip)' \
--vision_backbone 'UNET' \
--deep_supervision True \
--save_large_interval 100 \
--save_small_interval 100 \
--log_step_interval 100 \
--step_num 260000 100000 \
--warmup 10000 10000 \
--lr 1e-4 1e-5 \
--accumulate_grad_interval 1 \
--datasets_jsonl '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/challenge_data/train_10percent.jsonl' \
--dataset_config '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/dataset_config/cvpr25.json' \
--text_prompts_json '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/challenge_data/CVPR25_TextSegFMData_with_class.json' \
--text_encoder 'ours' \
--text_encoder_checkpoint '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-knowledge-pretraining/log/lp_cvpr25_debug/checkpoint/latest_step.pth' \
--text_encoder_partial_load True \
--open_bert_layer 12 \
--open_modality_embed False \
--num_workers 16 \
--max_queries 32 \
--crop_size 288 288 96 \
--patch_size 32 32 32 \
--batchsize_3d 2  \
--allow_repeat True \
--pin_memory False \
--nnUNet_aug True