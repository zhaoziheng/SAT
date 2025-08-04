import os
import random
import json
import traceback
import math
import warnings

from einops import rearrange, repeat, reduce
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import nibabel as nib

from train.dist import is_master

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False      
    
def split_3d(image_tensor, crop_size=[288, 288, 96]):
    # C H W D
    interval_h, interval_w, interval_d = crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2
    split_idx = []
    split_patch = []

    c, h, w, d = image_tensor.shape
    h_crop = max(math.ceil(h / interval_h) - 1, 1)
    w_crop = max(math.ceil(w / interval_w) - 1, 1)
    d_crop = max(math.ceil(d / interval_d) - 1, 1)

    for i in range(h_crop):
        h_s = i * interval_h
        h_e = h_s + crop_size[0]
        if  h_e > h:
            h_s = h - crop_size[0]
            h_e = h
            if h_s < 0:
                h_s = 0
        for j in range(w_crop):
            w_s = j * interval_w
            w_e = w_s + crop_size[1]
            if w_e > w:
                w_s = w - crop_size[1]
                w_e = w
                if w_s < 0:
                    w_s = 0
            for k in range(d_crop):
                d_s = k * interval_d
                d_e = d_s + crop_size[2]
                if d_e > d:
                    d_s = d - crop_size[2]
                    d_e = d
                    if d_s < 0:
                        d_s = 0
                split_idx.append([h_s, h_e, w_s, w_e, d_s, d_e])
                split_patch.append(image_tensor[:, h_s:h_e, w_s:w_e, d_s:d_e])
                
    return split_patch, split_idx 
    
class Evaluate_Dataset_OnlineCrop(Dataset):
    def __init__(self, loader, jsonl_file, max_queries=256, batch_size=2, patch_size=[288, 288, 96], evaluated_samples=set()):
        """
        max_queries: num of queries in a batch. can be very large.
        batch_size: num of image patch in a batch. be careful with this if you have limited gpu memory.
        evaluated_samples: to resume from an interrupted evaluation
        """
        self.loader = loader
        
        # load data info
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
            
        self.lines = []
        
        for sample in lines:
            # if resume and inherit medial results another evaluation
            sample_id = sample['patient_id']
            dataset_name = sample['dataset']
            if f'{dataset_name}_{sample_id}' not in evaluated_samples:
                self.lines.append(sample)
        
        self.max_queries = max_queries
        self.batch_size = batch_size
        self.patch_size = patch_size
        
        if is_master():          
            print(f'** Online Crop DATASET ** : Skip {len(lines)-len(self.lines)} samples, {len(self.lines)} to be evaluated')
            print(f'** Online Crop DATASET ** : Maximum {self.max_queries} queries, patch size {self.patch_size}')
        
    def __len__(self):
        return len(self.lines)
    
    def _split_labels(self, label_list):
        # split the labels into sub-lists
        if len(label_list) < self.max_queries:
            return [label_list], [[0, len(label_list)]]
        else:
            split_idx = []
            split_label = []
            query_num = len(label_list)
            n_crop = (query_num // self.max_queries + 1) if (query_num % self.max_queries != 0) else (query_num // self.max_queries)
            for n in range(n_crop):
                n_s = n*self.max_queries
                n_f = min((n+1)*self.max_queries, query_num)
                split_label.append(label_list[n_s:n_f])
                split_idx.append([n_s, n_f])
            return split_label, split_idx
    
    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'mr', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        else:
            return mod
        
    def _pad_if_necessary(self, patch):
        # NOTE: depth must be pad to 96
        b, c, h, w, d = patch.shape
        t_h, t_w, t_d = self.patch_size
        pad_in_h = 0 if h >= t_h else t_h - h
        pad_in_w = 0 if w >= t_w else t_w - w
        pad_in_d = 0 if d >= t_d else t_d - d
        if pad_in_h + pad_in_w + pad_in_d > 0:
            pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
            patch = F.pad(patch, pad, 'constant', 0)   # chwd
        return patch
        
    def __getitem__(self, idx):
        datum = self.lines[idx]
        sample_id = datum['patient_id']
        dataset_name = datum['dataset']
        
        with warnings.catch_warnings(): # ignore future warning from monai cropforeground
            warnings.filterwarnings("ignore", category=FutureWarning, message=r".*allow_smaller.*")
            img, mc_mask, labels, modality, image_path, mask_path = getattr(self.loader, dataset_name)(datum)
        
        if not isinstance(mc_mask, torch.FloatTensor):
            mc_mask = mc_mask.float()  # 转换为浮点类型
            
        patches, y1y2_x1x2_z1z2_ls = split_3d(img, crop_size=self.patch_size)
        
        # divide patches into batches
        batch_num = len(patches) // self.batch_size if len(patches) % self.batch_size == 0 else len(patches) // self.batch_size + 1
        batched_patches = []
        batched_y1y2_x1x2_z1z2 = []
        for i in range(batch_num):
            srt = i*self.batch_size
            end = min(i*self.batch_size+self.batch_size, len(patches))
            patch = torch.stack([patches[j] for j in range(srt, end)], dim=0)
            # NOTE: depth must be pad to 96
            patch = self._pad_if_necessary(patch)
            # for single-channel images, e.g. mri and ct, pad to 3
            # repeat sc image to mc
            if patch.shape[1] == 1:
                patch = repeat(patch, 'b c h w d -> b (c r) h w d', r=3)   
            batched_patches.append(patch) # b, *patch_size
            batched_y1y2_x1x2_z1z2.append([y1y2_x1x2_z1z2_ls[j] for j in range(srt, end)])

        # split labels into batches
        labels = datum['label']
        split_labels, split_n1n2 = self._split_labels(labels) # [xxx, ...] [[n1, n2], ...]
        modality = datum['modality']
        modality = self._merge_modality(modality.lower())
        for i in range(len(split_labels)):
            split_labels[i] = [label.lower() for label in split_labels[i]]
        
        assert mc_mask.shape[0] == len(labels), f'{mc_mask.shape[0]} != {len(labels)} for {datum["patient_id"]}'
        
        return {
            'dataset_name':datum['dataset'],
            'sample_id':sample_id, 
            'batched_patches':batched_patches, 
            'batched_y1y2_x1x2_z1z2':batched_y1y2_x1x2_z1z2, 
            'split_labels':split_labels, 
            'modality':modality,
            'split_n1n2':split_n1n2,
            'gt_segmentation':mc_mask,
            'labels':labels,
            'image_path':datum['image']
            }
        
def collate_fn(data):
    return data[0]
    