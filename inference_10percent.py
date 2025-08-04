import os
from glob import glob
import math

import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from einops import rearrange
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from einops import repeat
import scipy
from scipy.ndimage import zoom

from model.maskformer import Maskformer
from model.knowledge_encoder import Knowledge_Encoder

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

def pad_if_necessary(image):
    """
    Pad image to 288 288 96
    """
    c, h, w, d = image.shape
    croph, cropw, cropd = [288, 288, 96]
    pad_in_h = 0 if h >= croph else croph - h
    pad_in_w = 0 if w >= cropw else cropw - w
    pad_in_d = 0 if d >= cropd else cropd - d

    # Store padding information
    padding_info = (pad_in_h, pad_in_w, pad_in_d)

    if pad_in_h + pad_in_w + pad_in_d > 0:
        pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
        image = F.pad(image, pad, 'constant', 0)   # chwd

    return image, padding_info

def remove_padding(padded_image, padding_info):
    """
    Removes padding
    """
    pad_in_h, pad_in_w, pad_in_d = padding_info

    if len(padded_image.shape) == 4:
        if isinstance(padded_image, torch.Tensor):
            return padded_image[:, :padded_image.shape[1]-pad_in_h, :padded_image.shape[2]-pad_in_w, :padded_image.shape[3]-pad_in_d]
        else:  # numpy array
            return padded_image[:, :padded_image.shape[1]-pad_in_h, :padded_image.shape[2]-pad_in_w, :padded_image.shape[3]-pad_in_d]
    else:
        if isinstance(padded_image, torch.Tensor):
            return padded_image[:padded_image.shape[0]-pad_in_h, :padded_image.shape[1]-pad_in_w, :padded_image.shape[2]-pad_in_d]
        else:  # numpy array
            return padded_image[:padded_image.shape[0]-pad_in_h, :padded_image.shape[1]-pad_in_w, :padded_image.shape[2]-pad_in_d]

def respace_image(image: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray) -> np.ndarray:
    # Calculate zoom factors (ratio between current and target spacing)
    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    # Apply resampling using scipy.ndimage.zoom
    # order=1 uses linear interpolation
    resampled_image = scipy.ndimage.zoom(image, zoom_factors, order=1)
    return resampled_image

def read_npz_data(npz_file):

    data = np.load(npz_file, allow_pickle=True)

    raw_image = data['imgs'].astype(np.float32)  # 0~255
    raw_d, raw_h, raw_w = raw_image.shape
    # d h w -> h w d
    image = rearrange(raw_image, 'd h w -> h w d') # [h, w, d]

    # do respacing for CT
    if 'CT_' in npz_file:
        image = respace_image(image, data['spacing'], target_spacing=[1.0, 1.0, 3.0])

    # padding
    image = repeat(image, 'h w d -> c h w d', c=3)
    image = torch.tensor(image)
    image, padding_info = pad_if_necessary(image) # [3, h, w, d]
    _, h, w, d = image.shape

    text_prompts = data['text_prompts'].item()
    del text_prompts['instance_label']
    texts = list(text_prompts.values()) # ['xxx', ...]
    values = list(text_prompts.keys()) # [1, 2, ...]

    npz_file = os.path.basename(npz_file)
    if npz_file.startswith('CT'):
        modality = 'ct'
    elif npz_file.startswith('MR'):
        modality = 'mri'
    elif npz_file.startswith('US'):
        modality = 'us'
    elif npz_file.startswith('PET'):
        modality = 'pet'
    elif npz_file.startswith('Microscopy'):
        modality = 'microscopy'
    else:
        raise ValueError(f"Unknown modality for file {npz_file}")

    patches, y1y2_x1x2_z1z2_ls = split_3d(image, crop_size=[288, 288, 96])    # [[3, 288, 288, 96], ...]  # [[y1, y2, x1, x2, z1, z2], ...]

    return {
        'npz_file_name': npz_file,
        'modality': modality,
        'texts': texts,
        'values': values,
        'original_shape': (raw_h, raw_w, raw_d),
        'current_shape': (h, w, d),
        'patches': patches,
        'y1y2_x1x2_z1z2_ls': y1y2_x1x2_z1z2_ls,
        'padding_info': padding_info,
        'raw_image': raw_image
    }

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

def main():

    # set gpu
    device=torch.device("cuda", 0)

    # load model
    model = Maskformer('UNET', [288, 288, 96], [32, 32, 32], False)
    model = model.to(device)
    checkpoint = torch.load('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/nano_CVPR2025_w_pretrain(nnunet_aug_w_flip)(woNorm)/checkpoint/step_150000.pth', map_location=device)
    # Remove 'module.' prefix from keys in checkpoint
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if 'mid_mask_embed_proj' in key:
            continue
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value  # Remove first 7 chars ('module.')
        else:
            new_state_dict[key] = value
    checkpoint['model_state_dict'] = new_state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # load text encoder
    text_encoder = Knowledge_Encoder()
    text_encoder = text_encoder.to(device)
    checkpoint = torch.load('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/nano_CVPR2025_w_pretrain(nnunet_aug_w_flip)(woNorm)/checkpoint/text_encoder_step_150000.pth', map_location=device)
    # Remove 'module.' prefix from keys in checkpoint
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value  # Remove first 7 chars ('module.')
        else:
            new_state_dict[key] = value
    checkpoint['model_state_dict'] = new_state_dict
    text_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # begin inference
    model.eval()
    text_encoder.eval()
    with torch.no_grad():
        
        # gaussian kernel to accumulate predcition
        gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)    # hwd
        
        for npz_file in glob('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/Version2/3D_val_npz/*.npz'):
        
            # load and process inference data
            data_dict = read_npz_data(npz_file)
            
            # Extract individual values from dictionary
            file_name = data_dict['npz_file_name']
            modality = data_dict['modality']
            
            text_prompts = data_dict['texts']
            label_values = data_dict['values']
            
            original_shape = data_dict['original_shape']
            current_shape = data_dict['current_shape']
            batched_patches = data_dict['patches']
            batched_y1y2_x1x2_z1z2 = data_dict['y1y2_x1x2_z1z2_ls']
            padding_info = data_dict['padding_info']
            raw_image = data_dict['raw_image']
            
            modality_code_dict = {
                    'ct':0,
                    'mri':1,
                    'us':2,
                    'pet':3,
                    'microscopy':4
                }
            modality_code = torch.tensor([modality_code_dict[modality]]).to(device)
            
            h, w, d = current_shape
            n = len(text_prompts)
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))
            with autocast():
                
                # encode text prompts
                queries = text_encoder(text_prompts, modality_code)   # convert text prompts to embeds
                torch.cuda.empty_cache()
                
                # for each batch of patches, query with all labels
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patches, batched_y1y2_x1x2_z1z2):   # [c, h, w, d]
                    patches = patches.unsqueeze(0).to(device=device)    # [b, c, h, w, d]
                    prediction_patch = model(queries=queries, image_input=patches, train_mode=False)
                    prediction_patch = torch.sigmoid(prediction_patch)  # bnhwd
                    prediction_patch = prediction_patch.detach() # .cpu().numpy()
                
                    # fill in 
                    y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls
                    # gaussian accumulation
                    tmp = prediction_patch[0, :, :y2-y1, :x2-x1, :z2-z1] * gaussian[:y2-y1, :x2-x1, :z2-z1] # on gpu
                    prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                    accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2-y1, :x2-x1, :z2-z1].cpu()
                
                # avg            
                prediction = prediction / accumulation
                prediction = torch.where(prediction>0.5, 1.0, 0.0)
                prediction = prediction.numpy()

            # save prediction
            results = np.zeros((h, w, d)) # hwd
            for j, (text, value) in enumerate(zip(text_prompts, label_values)):
                results += prediction[j, :, :, :] * int(value)
            results = remove_padding(results, padding_info)
            # Check if the current shape is different from original shape (respaced) and resize if needed
            current_h, current_w, current_d = results.shape
            original_h, original_w, original_d = original_shape
            if current_h != original_h or current_w != original_w or current_d != original_d:
                # Use scipy's resize function to restore to original shape
                zoom_factors = (original_h/current_h, original_w/current_w, original_d/current_d)
                # Use nearest neighbor interpolation (order=0) to preserve label values
                results = zoom(results, zoom_factors, order=0)
                print(f"Resized segmentation from {(current_h, current_w, current_d)} to {(original_h, original_w, original_d)}")
            results = rearrange(results, 'h w d -> d h w')
            np.savez_compressed(os.path.join("/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/Version2/3D_val_10percent_prediction", file_name), segs=results)

if __name__ == '__main__':
    main()
    
    
        
    
    