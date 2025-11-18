import os
import time

import torch
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import shutil
from scipy.ndimage import gaussian_filter

from train.dist import is_master

def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def inference(model, text_encoder, device, testset, testloader, nib_dir):
    # collate in master process
    if is_master():
        jsonl_file = testset.jsonl_file.split('/')[-1]
        shutil.copy(testset.jsonl_file, f'{nib_dir}/{jsonl_file}')
    
    model.eval()
    text_encoder.eval()
        
    with torch.no_grad():
        
        data_time = 0
        pred_time = 0
        
        avg_patch_batch_num = 0
        avg_query_batch_num = 0
        
        # in ddp, only master process display the progress bar
        if is_master():
            testloader = tqdm(testloader, disable=False)
        else:
            testloader = tqdm(testloader, disable=True)  
            
        # gaussian kernel to accumulate predcition
        gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)    # hwd
        
        end_time = time.time()
        for batch in testloader:
            # data loading
            dataset_name = batch['dataset_name']
            sample_id = batch['sample_id'] 
            batched_patches = batch['batched_patches']
            batched_y1y2_x1x2_z1z2 = batch['batched_y1y2_x1x2_z1z2']
            split_labels = batch['split_queries'] 
            split_n1n2 = batch['split_n1n2']
            labels = batch['labels']
            modality = batch['modality']
            
            _, h, w, d = batch['chwd']
            n = len(labels)
            
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))
            
            data_time += (time.time()-end_time) 
            end_time = time.time()
            
            avg_patch_batch_num += len(batched_patches)
            avg_query_batch_num += len(split_labels)
            
            with autocast():
                
                # for each batch of queries
                queries_ls = []
                for labels_ls, n1n2 in zip(split_labels, split_n1n2):  # convert list of texts to list of embeds
                    queries_ls.append(text_encoder(labels_ls, modality))
                    
                torch.cuda.empty_cache()
                      
                # for each batch of patches, query with all labels
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patches, batched_y1y2_x1x2_z1z2):   # [b, c, h, w, d]
                    patches = patches.to(device=device)
                    prediction_patch = model(queries=queries_ls, image_input=patches)
                    prediction_patch = torch.sigmoid(prediction_patch)  # bnhwd
                    prediction_patch = prediction_patch.detach() # .cpu().numpy()
                    
                    # fill in 
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]

                        # gaussian accumulation
                        tmp = prediction_patch[b, :, :y2-y1, :x2-x1, :z2-z1] * gaussian[:y2-y1, :x2-x1, :z2-z1] # on gpu
                        prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                        accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2-y1, :x2-x1, :z2-z1].cpu()
                            
            # avg            
            prediction = prediction / accumulation
            prediction = torch.where(prediction>0.5, 1.0, 0.0)
            prediction = prediction.numpy()
            
            pred_time += (time.time()-end_time)
            end_time = time.time()
            
            # visualization  
            Path(f'{nib_dir}/{dataset_name}').mkdir(exist_ok=True, parents=True)
            # 将image、gt和prediction保存下来
            results = np.zeros((h, w, d)) # hwd
            for j, label in enumerate(labels):
                results += prediction[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
                Path(f'{nib_dir}/{dataset_name}/seg_{sample_id}').mkdir(exist_ok=True, parents=True)
                # 每个label单独一个nii.gz
                segobj = nib.nifti2.Nifti1Image(prediction[j, :, :, :], np.eye(4))
                nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{sample_id}/{label}.nii.gz')
                
            segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
            nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{sample_id}.nii.gz')
            
            image = batch['image'].numpy()
            if image.ndim == 4:
                image = image[0, :, :, :]   # h w d
            imgobj = nib.nifti2.Nifti1Image(image, np.eye(4))
            nib.save(imgobj, f'{nib_dir}/{dataset_name}/img_{sample_id}.nii.gz')
                
        torch.cuda.empty_cache()    