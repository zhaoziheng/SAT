#!/bin/bash
#SBATCH --job-name=eval_pro
#SBATCH --quotatype=auto
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=128G
#SBATCH --chdir=/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/sbatch
#SBATCH --output=/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/sbatch/%x-%j.out
#SBATCH --error=/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/sbatch/%x-%j.error
###SBATCH -w SH-IDC1-10-140-0-[...], SH-IDC1-10-140-1-[...]
###SBATCH -x SH-IDC1-10-140-0-[...], SH-IDC1-10-140-1-[...]

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
--nproc_per_node 1 \
--rdzv_id 100 \
--rdzv_backend c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/evaluate.py \
--rcd_dir 'your_rcd_dir' \
--rcd_file 'your_rcd_file_name' \
--resume False \
--visualization False \
--deep_supervision False \
--datasets_jsonl 'jsonl generated from SAT-DS Step 4' \
--crop_size 288 288 96 \
--online_crop True \
--vision_backbone 'UNET-L' \
--checkpoint 'your ckpt' \
--partial_load True \
--text_encoder 'ours' \
--text_encoder_checkpoint 'your text encoder ckpt' \
--batchsize_3d 2 \
--max_queries 256 \
--pin_memory False \
--num_workers 4 \
--dice True \
--nsd True