#!/bin/bash
#SBATCH --job-name=sat_nano
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=256G
#SBATCH --chdir=logs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.error
###SBATCH --exclude=xxx

export NCCL_DEBUG=INFO
export NCCL_IBEXT_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
echo NODELIST=${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
MASTER_PORT=$((RANDOM % 101 + 20000))
echo "MASTER_ADDR="$MASTER_ADDR

srun torchrun \
--nnodes 1 \
--nproc_per_node 8 \
--rdzv_id 100 \
--rdzv_backend c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py \
--log_dir 'log' \
--name 'sat_nano' \
--vision_backbone 'UNET' \
--deep_supervision True \
--save_large_interval 10000 \
--save_small_interval 1000 \
--log_step_interval 1000 \
--step_num 200000 \
--warmup 20000 \
--lr 1e-4 \
--accumulate_grad_interval 1 \
--datasets_jsonl 'trainset.jsonl' \
--dataset_config 'data/dataset_config/72.json' \
--text_encoder 'ours' \
--text_encoder_checkpoint 'checkpoint/text_encoder_checkpoint.pth' \
--text_encoder_partial_load False \
--open_bert_layer 12 \
--open_modality_embed True \
--num_workers 8 \
--max_queries 32 \
--crop_size 288 288 64 \
--patch_size 32 32 32 \
--batchsize_3d 2  \
--allow_repeat True