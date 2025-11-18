import torch
import torch.nn as nn
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from .maskformer import Maskformer

from train.dist import is_master


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def build_maskformer(args, device, gpu_id):
    model = Maskformer(args.vision_backbone, args.crop_size, args.patch_size, args.deep_supervision)

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if is_master():
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters")
            
    return model


def load_checkpoint(checkpoint, 
                    resume, 
                    partial_load, 
                    model, 
                    device,
                    optimizer=None,
                    ):
    
    if is_master():
        print('** CHECKPOINT ** : Load checkpoint from %s' % (checkpoint))
    
    checkpoint = torch.load(checkpoint, map_location=device)
        
    # load part of the checkpoint
    if partial_load:
        model_dict =  model.state_dict()
        # check difference
        unexpected_state_dict = [k for k in checkpoint['model_state_dict'].keys() if k not in model_dict.keys()]
        missing_state_dict = [k for k in model_dict.keys() if k not in checkpoint['model_state_dict'].keys()]
        unmatchd_state_dict = [k for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
        # load partial parameters
        state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        if is_master():
            print('The following parameters are unexpected in SAT checkpoint:\n', unexpected_state_dict)
            print('The following parameters are missing in SAT checkpoint:\n', missing_state_dict)
            print('The following parameters have different shapes in SAT checkpoint:\n', unmatchd_state_dict)
            print('The following parameters are loaded in SAT:\n', state_dict.keys())
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # if resume, load optimizer and step
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = int(checkpoint['step']) + 1
    else:
        start_step = 1
        
    return model, optimizer, start_step


def inherit_knowledge_encoder(knowledge_encoder_checkpoint,
                              model,
                              device
                              ):
    # inherit unet encoder and multiscale feature projection layer from knowledge encoder
    checkpoint = torch.load(knowledge_encoder_checkpoint, map_location=device)
        
    model_dict =  model.state_dict()
    visual_encoder_state_dict = {k.replace('atlas_tower', 'backbone'):v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower.encoder' in k}    # encoder部分
    model_dict.update(visual_encoder_state_dict)
    proj_state_dict = {k.replace('atlas_tower.', ''):v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower.projection_layer' in k}    # projection layer部分
    model_dict.update(proj_state_dict)
    model.load_state_dict(model_dict)
    
    if is_master():
        print('** CHECKPOINT ** : Inherit pretrained unet encoder from %s' % (knowledge_encoder_checkpoint))
        print('The following parameters are loaded in SAT:\n', list(visual_encoder_state_dict.keys())+list(proj_state_dict.keys()))
        
    return model