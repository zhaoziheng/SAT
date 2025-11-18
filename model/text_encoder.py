import torch
import numpy as np
import os
import torch.nn as nn

from einops import rearrange, reduce, repeat

from .knowledge_encoder import Knowledge_Encoder
from .med_cpt import MedCPT
from .base_bert import BaseBERT
from train.dist import is_master

def compute_average_gradient(module):
    # 初始化梯度总和和参数计数
    total_gradient = 0.0
    total_params = 0
    
    # 遍历module的所有参数
    for param in module.parameters():
        if param.grad is not None:
            # 累加此参数的梯度绝对值
            total_gradient += param.grad.abs().mean().item()
            total_params += 1
    
    # 计算平均梯度
    if total_params > 0:
        average_gradient = total_gradient / total_params
    else:
        average_gradient = None
    
    return average_gradient

class Text_Encoder(nn.Module):
    def __init__(self, 
                 text_encoder,
                 checkpoint=None,
                 # other params
                 open_bert_layer=12,
                 open_modality_embed=False,
                 partial_load=False,
                 gpu_id=None,
                 device=None):
        super().__init__()

        self.device = device
        
        # choose text encoder
        class_name = {
            'ours': Knowledge_Encoder,
            'medcpt': MedCPT,
            'basebert': BaseBERT,
        }[text_encoder]
        
        model = class_name()
        model = model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        
        # load checkpoint
        if checkpoint:
            if is_master():
                print(f"** QUERY ** Load encoder from {checkpoint}.")
                
            checkpoint = torch.load(checkpoint, map_location=device)
            checkpoint['model_state_dict'] = {k:v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower' not in k and 'temperature' not in k}
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
                    print('The following parameters are unexpected in query generator checkpoint:\n', unexpected_state_dict)
                    print('The following parameters are missing in query generator checkpoint:\n', missing_state_dict)
                    print('The following parameters have different shapes in query generator checkpoint:\n', unmatchd_state_dict)
                    print('The following parameters are loaded in query generator :\n', state_dict.keys())
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
        # open bert
        for name, param in model.named_parameters():
            if 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and ('pooler' in name or 'mlp_embed' in name):
                param.requires_grad = True
            elif open_modality_embed and 'modality_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        self.model = model

    def forward(self, label_name, modality_name):
        """
        Args:
            label_name (List of List of Str / List of Str): B x N / N
            modality_name (List / Str): B / 1
            NOTE: a list of labels paired with one modality
            
        Return:
            queries (Tensor): B x N / N
        """
        if isinstance(label_name[0], list):
            batch_size = len(label_name)
            num_query = len(label_name[0])
            input_text = [t for t_ls in label_name for t in t_ls]    # BN
            modality = [mod for mod in modality_name for n in range(num_query)] # repeat each mod for N times -> BN
        else:
            num_query = len(label_name)
            input_text = label_name  # N
            modality = [modality_name for n in range(num_query)]   # N
            
        # name to code
        modality_code_dict = {
                'ct':0,
                'mri':1,
                'us':2,
                'pet':3,
            }
        modality_code = torch.tensor([modality_code_dict[mod] for mod in modality])   # bn
            
        # get embed
        queries = self.model(input_text, modality_code)
        
        if isinstance(label_name[0], list):
            queries = rearrange(queries, '(b n) d -> b n d', b=batch_size, n=num_query)

        return queries

        
        
        
        