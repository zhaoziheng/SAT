import numpy as np
import torch
from einops import repeat, rearrange
import os
import json
import scipy

import re

def respace_image(image: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray) -> np.ndarray:
    # Calculate zoom factors (ratio between current and target spacing)
    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    # Apply resampling using scipy.ndimage.zoom
    # order=1 uses linear interpolation
    resampled_image = scipy.ndimage.zoom(image, zoom_factors, order=1)
    return resampled_image

def MR_CHAOS_T2(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    return image, mask, spacing

def MR_WMH_T1(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    return image, mask, spacing

def MR_LeftAtrium(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    return image, mask, spacing

def MR_ISLES_ADC(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_WMH_FLAIR(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    return image, mask, spacing

def MR_TotalSeg(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    return image, mask, spacing

def MR_CervicalCancer(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def MR_AMOS(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    return image, mask, spacing

def MR_ProstateADC(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_ProstateT2(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_QIN_PROSTATE_Lesion(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_Heart_ACDC(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    return image, mask, spacing

def MR_HNTS_MRG_HeadTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    return image, mask, spacing

def MR_ISLES_DWI(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_CHAOS_T1(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    return image, mask, spacing

def nucmm(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def Microscopy_SELMA3D_ADplaques(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def Microscopy_SELMA3D_vessel(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def Microscopy_SELMA3D_neural_activity_marker(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def Microscopy_SELMA3D_nuceus(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def autoPET(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def US_Low_limb_Leg(file_name, image, mask, spacing):
    image = respace_image(image, spacing, [1, 1, 1])
    mask = respace_image(mask, spacing, [1, 1, 1])
    return image, mask, [1, 1, 1]

def US_Cardiac(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    return image, mask, spacing

def CT_LungLesion(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_AMOS(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_LiverTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_Abdomen1K(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_PancreasTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    if 'CT_Lesion_PANORAMA_' in file_name:
        # orientation
        image = image[:,:,::-1]
        mask = mask[:,:,::-1]
        image = np.rot90(image, k=-1, axes=(0, 1))
        image = image[::-1,:,:]
        mask = np.rot90(mask, k=-1, axes=(0, 1))
        mask = mask[::-1,:,:]
    if 'CT_AbdTumor_pancreas_' in file_name:
        # orientation
        image = np.rot90(image, k=-1, axes=(0, 1))
        image = image[:,::-1,:]
        mask = np.rot90(mask, k=-1, axes=(0, 1))
        mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_TotalSeg_cardiac(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_Lungs(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_ThoracicOrgans_TCIA_LCTSC(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_AdrenalTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_WholeBodyTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_TotalSeg_organs(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_SegRap_HeadNeckTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_ColonTumor(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_LymphNode(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_HaN_Seg(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_AirwayTree(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_TotalSeg_vertebrae(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_COVID19_Infection(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_AbdomenAtlas(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_Aorta(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[::-1,:,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[::-1,:,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def CT_TotalSeg_muscles(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    image = image[:,::-1,:]
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    mask = mask[:,::-1,:]
    # respace
    image = respace_image(image, spacing, [1, 1, 3])
    mask = respace_image(mask, spacing, [1, 1, 3])
    return image, mask, spacing

def Microscopy_urocell_Endolysosomes(file_name, image, mask, spacing):
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def cremi(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    image = respace_image(image, spacing, [8, 8, 40])
    mask = respace_image(mask, spacing, [8, 8, 40])
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def Microscopy_urocell_Mitochondria(file_name, image, mask, spacing):
    # mask
    mask = np.where(mask>0, 1.0, 0.0)
    return image, mask, spacing

def MR_BraTS_T1n(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_BraTS_T1c(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_BraTS_T2w(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_BraTS_T2f(file_name, image, mask, spacing):
    # 找到image中最小维度的位置，并将该维度移到最后
    order = list(range(image.ndim))
    min_dim = int(np.argmin(image.shape))
    order.pop(min_dim)
    order.append(min_dim)
    image = np.transpose(image, order)
    mask = np.transpose(mask, order)
    # orientation
    image = np.rot90(image, k=1, axes=(0, 1))
    mask = np.rot90(mask, k=1, axes=(0, 1))
    return image, mask, spacing

def MR_T1c_crossMoDA_Tumor_Cochlea(file_name, image, mask, spacing):
    # d h w -> h w d
    image = rearrange(image, 'd h w -> h w d')
    mask = rearrange(mask, 'd h w -> h w d')
    # orientation
    image = np.rot90(image, k=-1, axes=(0, 1))
    mask = np.rot90(mask, k=-1, axes=(0, 1))
    return image, mask, spacing

NAME2LOADER = {
    'MR_T1c_crossMoDA_Tumor_Cochlea': MR_T1c_crossMoDA_Tumor_Cochlea,
    'MR_BraTS-T1n': MR_BraTS_T1n,
    'MR_BraTS-T1c': MR_BraTS_T1c,
    'MR_BraTS-T2w': MR_BraTS_T2w,
    'MR_BraTS-T2f': MR_BraTS_T2f,
    'MR_CHAOS-T2': MR_CHAOS_T2,
    'MR_WMH_T1': MR_WMH_T1,
    'MR_LeftAtrium': MR_LeftAtrium,
    'MR_ISLES_ADC': MR_ISLES_ADC,
    'MR_WMH_FLAIR': MR_WMH_FLAIR,
    'MR_TotalSeg': MR_TotalSeg,
    'MR_CervicalCancer': MR_CervicalCancer,
    'MR_AMOS': MR_AMOS,
    'MR_ProstateADC': MR_ProstateADC,
    'MR_ProstateT2': MR_ProstateT2,
    'MR_QIN-PROSTATE-Lesion': MR_QIN_PROSTATE_Lesion,
    'MR_Heart_ACDC': MR_Heart_ACDC,
    'MR_HNTS-MRG_HeadTumor': MR_HNTS_MRG_HeadTumor,
    'MR_ISLES_DWI': MR_ISLES_DWI,
    'MR_CHAOS-T1': MR_CHAOS_T1,
    'nucmm': nucmm,
    'cremi': cremi,
    'Microscopy_SELMA3D_ADplaques': Microscopy_SELMA3D_ADplaques,
    'Microscopy_SELMA3D_vessel': Microscopy_SELMA3D_vessel,
    'Microscopy_SELMA3D_neural_activity_marker': Microscopy_SELMA3D_neural_activity_marker,
    'Microscopy_SELMA3D_nuceus': Microscopy_SELMA3D_nuceus,
    'Microscopy_urocell_Endolysosomes': Microscopy_urocell_Endolysosomes,
    'Microscopy_urocell_Mitochondria': Microscopy_urocell_Mitochondria,
    'autoPET': autoPET,
    'US_Low-limb-Leg': US_Low_limb_Leg,
    'US_Cardiac': US_Cardiac,
    'CT_LungLesion': CT_LungLesion,
    'CT_AMOS': CT_AMOS,
    'CT_LiverTumor': CT_LiverTumor,
    'CT_Abdomen1K': CT_Abdomen1K,
    'CT_PancreasTumor': CT_PancreasTumor,
    'CT_TotalSeg_cardiac': CT_TotalSeg_cardiac,
    'CT_Lungs': CT_Lungs,
    'CT_ThoracicOrgans-TCIA-LCTSC': CT_ThoracicOrgans_TCIA_LCTSC,
    'CT_AdrenalTumor': CT_AdrenalTumor,
    'CT_WholeBodyTumor': CT_WholeBodyTumor,
    'CT_TotalSeg_organs': CT_TotalSeg_organs,
    'CT_SegRap_HeadNeckTumor': CT_SegRap_HeadNeckTumor,
    'CT_ColonTumor': CT_ColonTumor,
    'CT_LymphNode': CT_LymphNode,
    'CT_HaN-Seg': CT_HaN_Seg,
    'CT_AirwayTree': CT_AirwayTree,
    'CT_TotalSeg-vertebrae': CT_TotalSeg_vertebrae,
    'CT_COVID19-Infection': CT_COVID19_Infection,
    'CT_AbdomenAtlas': CT_AbdomenAtlas,
    'CT_Aorta': CT_Aorta,
    'CT_TotalSeg_muscles': CT_TotalSeg_muscles,
}