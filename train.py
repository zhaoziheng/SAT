import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.distributed as dist

from data.build_dataset import build_dataset

from model.build_model import build_maskformer, load_checkpoint, inherit_knowledge_encoder
from model.text_encoder import Text_Encoder

from train.params import parse_args
from train.logger import set_up_log
from train.loss import BinaryDiceLoss
from train.scheduler import cosine_lr
from train.trainer import Trainer
from train.dist import is_master

def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    # get configs
    args = parse_args()
    
    # set logger
    if is_master():
        checkpoint_dir, tb_writer, log_file = set_up_log(args)
    else:
        checkpoint_dir = None
        tb_writer = None
        log_file = None
        
    # set random seed for reproducibility
    # set_seed(args)
    
    # set up distribution (identify the device of current process
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    # dispaly
    if is_master():
        print('** GPU NUM ** : ', torch.cuda.device_count())  # 打印gpu数量
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    rank = dist.get_rank()
    print(f"** DDP ** : Start running DDP on rank {rank}.")
    
    # dataset and loader
    trainset, trainloader, sampler = build_dataset(args)
    
    # set model (by default gpu
    model = build_maskformer(args, device, gpu_id)
    
    # build, load and set trainer parameters in knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=args.open_bert_layer,
        open_modality_embed=args.open_modality_embed,
        gpu_id=gpu_id,
        device=device
    )
    
    # set loss calculator
    dice_loss = BinaryDiceLoss(reduction='none')
    bce_w_logits_loss = nn.BCEWithLogitsLoss(reduction='none') # safe for amp
    
    # set optimizer 
    target_parameters = list(model.parameters()) + list(text_encoder.parameters())
    optimizer = optim.AdamW(
        target_parameters,
        lr=args.lr[0],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    
    # set scheduler
    total_steps = args.step_num
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    # if restart cosine annealing, total_steps = sum of steps in each stage
    if isinstance(total_steps, list):
        total_steps = sum(total_steps)
    
    # load checkpoint and set trainable parameters
    start_step = 1  # the real steps we have gone
    if args.checkpoint:
        model, optimizer, start_step = load_checkpoint(
            checkpoint=args.checkpoint,
            resume=args.resume,
            partial_load=args.partial_load,
            model=model, 
            optimizer=optimizer, 
            device=device,
        )
    elif args.inherit_knowledge_encoder:   # inherit the unet encoder in pretraining
        model = inherit_knowledge_encoder(
            knowledge_encoder_checkpoint=args.knowledge_encoder_checkpoint,
            model=model,
            device=device,
        )
    if is_master():
        print(f'Starting from step {start_step}')

    # check untrainable parameters
    if is_master():
        print('The following parameters in SAT are frozen:')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)
        print('The following parameters in text encoder are frozen:')
        for name, param in text_encoder.named_parameters():
            if not param.requires_grad:
                print(name)
            
    trainer = Trainer(
                    args=args,
                    model=model,
                    text_encoder=text_encoder,
                    device=device,
                    trainset=trainset,
                    trainloader=trainloader,
                    sampler=sampler,
                    dice_loss=dice_loss,
                    bce_w_logits_loss=bce_w_logits_loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tb_writer=tb_writer,
                    checkpoint_dir=checkpoint_dir,
                    log_file=log_file
                    )
        
    for step in range(start_step, total_steps+1): 
        
        # make sure the train is not interrupted
        if is_master() and step%10==0:
            print(f'Training Step %d'%step)
        
        # accmulate grad
        for accum in range(args.accumulate_grad_interval):
            
            trainer.train_one_step(step)
        
if __name__ == '__main__':
    main()
    
    
        
    
    