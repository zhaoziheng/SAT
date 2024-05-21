import os
import random
import traceback
import json
import math

from einops import rearrange, repeat, reduce
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib
import monai

from train.dist import is_master

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

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False  
    
def Normalization(torch_image, image_type):
    # rgb_list = ['rgb', 'photograph', 'laparoscopy', 'colonoscopy', 'microscopy', 'dermoscopy', 'fundus', 'fundus image']
    np_image = torch_image.numpy()
    if image_type.lower() == 'ct':
        lower_bound, upper_bound = -500, 1000
        np_image = np.clip(np_image, lower_bound, upper_bound)
        np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    else:
        lower_bound, upper_bound = np.percentile(np_image, 0.5), np.percentile(np_image, 99.5)
        np_image = np.clip(np_image, lower_bound, upper_bound)
        np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    return torch.tensor(np_image)     

def load_image(datum):
    orientation_code = datum['orientation_code'] if 'orientation_code' in datum else "RAS"
    
    monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes=orientation_code, keys=['image']),   # zyx
                monai.transforms.Spacingd(keys=["image"], pixdim=(1, 1, 3), mode=("bilinear")),
                monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.ToTensord(keys=["image"]),
            ]
        )
    dictionary = monai_loader({'image':datum['image']})
    img = dictionary['image']
    img = Normalization(img, datum['modality'].lower())
    
    return img, datum['label'], datum['modality'], datum['image']
        
class Inference_Dataset(Dataset):
    def __init__(self, jsonl_file, max_queries=256, batch_size=2, patch_size=[288, 288, 96]):
        """
        max_queries: num of queries in a batch. can be very large.
        batch_size: num of image patch in a batch. be careful with this if you have limited gpu memory.
        evaluated_samples: to resume from an interrupted evaluation
        """
        # load data info
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        self.lines = [json.loads(line) for line in lines]
        
        self.max_queries = max_queries
        self.batch_size = batch_size
        self.patch_size = patch_size
        
        if is_master():          
            print(f'** DATASET ** : load {len(lines)} samples')
        
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
        img, labels, modality, image_path = load_image(datum)
        c,h,w,d = img.shape
        
        # image to patches
        patches, y1y2_x1x2_z1z2_ls = split_3d(img, crop_size=[288, 288, 96])
        
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
        split_labels, split_n1n2 = self._split_labels(labels) # [xxx, ...] [[n1, n2], ...]
        modality = self._merge_modality(modality.lower())
        for i in range(len(split_labels)):
            split_labels[i] = [label.lower() for label in split_labels[i]]

        # the unique id of sample, used to name output
        sample_id = image_path.split('/')[-1].replace('.nii.gz', '')    # 0.nii.gz -> 0
        
        return {
            'dataset_name':datum['dataset'],
            'sample_id':sample_id, 
            'image':img,
            'batched_patches':batched_patches, 
            'batched_y1y2_x1x2_z1z2':batched_y1y2_x1x2_z1z2, 
            'split_queries':split_labels, 
            'split_n1n2':split_n1n2,
            'labels':labels,
            'chwd':[c,h,w,d],
            'modality':modality,
            }
        
def collate_fn(data):
    return data[0]