import os
import random
import math
import monai

from einops import rearrange, repeat, reduce
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False         
    
class Inference_Dataset(Dataset):
    def __init__(self, 
                 jsonl_file,
                 patch_size=[288,288,96], 
                 max_queries=16,
                 batch_size=2):
        """
        Dataset for inference.
        
        Args:
            json_file (str): path to a jsonl containing data samples
            patch_size (int, optional): size of patch. Defaults to [256,256,96].
            max_queries (int, optional): maximum text query in a forward. Defaults to 16.
            batch_size (int, optional): num of patch in a forward. Defaults to 16.
        """
        self.patch_size = patch_size
        self.max_queries = max_queries
        self.batch_size = batch_size
        
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line) for line in lines]
                
    def __len__(self):
        return len(self.data)
    
    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'flair', 'dwi', 'mr', 'magnetic resonance', 'magnetic resonance imaging']):
            return 'mri'
        if contains(mod, ['ct', 'computed tomography']):
            return 'ct'
    
    def _normalization(self, torch_image, image_type):
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

    def _load_data(self, datum:dict) -> tuple:
        # load nii.gz data
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image']),  # by default axial plane
                monai.transforms.Spacingd(keys=["image"], pixdim=(1, 1, 3), mode=("bilinear")),
                monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.ToTensord(keys=["image"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image']})
        img = dictionary['image']
        
        modality = self._merge_modality(datum['modality'].lower())
        img = self._normalization(img, modality)
        img = (img-img.min())/(img.max()-img.min())
        
        return img, datum['label'], modality, datum['image']
    
    def _split_labels(self, label_list):
        # split the labels into sub-lists
        if len(label_list) < self.max_queries:
            return [label_list], [[0, len(label_list)]]
        else:
            split_idx = []
            split_label = []
            query_num = len(label_list)
            n_crop = query_num // self.max_queries + 1
            for n in range(n_crop):
                n_s = n*self.max_queries
                n_f = min((n+1)*self.max_queries, query_num)
                split_label.append(label_list[n_s:n_f])
                split_idx.append([n_s, n_f])
            return split_label, split_idx
        
    def _split_3d(self, image_tensor, patch_size=[288, 288, 96]):
        # split image into patches
        interval_h, interval_w, interval_d = patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2
        split_idx = []
        split_patch = []

        c, h, w, d = image_tensor.shape
        h_crop = max(math.ceil(h / interval_h) - 1, 1)
        w_crop = max(math.ceil(w / interval_w) - 1, 1)
        d_crop = max(math.ceil(d / interval_d) - 1, 1)

        for i in range(h_crop):
            h_s = i * interval_h
            h_e = h_s + patch_size[0]
            if  h_e > h:
                h_s = h - patch_size[0]
                h_e = h
                if h_s < 0:
                    h_s = 0
            for j in range(w_crop):
                w_s = j * interval_w
                w_e = w_s + patch_size[1]
                if w_e > w:
                    w_s = w - patch_size[1]
                    w_e = w
                    if w_s < 0:
                        w_s = 0
                for k in range(d_crop):
                    d_s = k * interval_d
                    d_e = d_s + patch_size[2]
                    if d_e > d:
                        d_s = d - patch_size[2]
                        d_e = d
                        if d_s < 0:
                            d_s = 0
                    split_idx.append([h_s, h_e, w_s, w_e, d_s, d_e])
                    split_patch.append(image_tensor[:, h_s:h_e, w_s:w_e, d_s:d_e])
                    
        return split_patch, split_idx
    
    def _pad_if_necessary(self, patch):
        # pad to patch size
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
        image, labels, modality, image_path = self._load_data(self.data[idx])
        
        # split into patches
        patch_ls, y1y2x1x2z1z2_ls = self._split_3d(image, self.patch_size)
        # group patches into batches
        batch_num = len(patch_ls) // self.batch_size if len(patch_ls) % self.batch_size == 0 else len(patch_ls) // self.batch_size + 1
        batched_patches = []
        batched_y1y2_x1x2_z1z2 = []
        for i in range(batch_num):
            srt = i*self.batch_size
            end = min(i*self.batch_size+self.batch_size, len(patch_ls))
            patch = torch.stack([patch_ls[j] for j in range(srt, end)], dim=0)
            patch = self._pad_if_necessary(patch)
            # for single-channel images, e.g. mri and ct, pad to 3
            if patch.shape[1] == 1:
                patch = repeat(patch, 'b c h w d -> b (c r) h w d', r=3)   
            batched_patches.append(patch) # b, *patch_size
            batched_y1y2_x1x2_z1z2.append([y1y2x1x2z1z2_ls[j] for j in range(srt, end)])

        # split queries
        split_label, split_n1n2 = self._split_labels(labels)
        
        # generate prompts
        split_prompt = []
        for label_ls in split_label:
            split_prompt.append([f'Modality:{modality}, Plane:Unknown, Region:Unknown, Anatomy Name:{label.lower()}' for label in label_ls])
        
        return {
            'image':image, 
            'batched_patch':batched_patches,
            'batched_y1y2x1x2z1z2':batched_y1y2_x1x2_z1z2,
            'labels':labels, 
            'split_n1n2':split_n1n2,
            'split_prompt':split_prompt,
            'image_path':image_path,
            }
        
def inference_collate_fn(data):
    return data[0]