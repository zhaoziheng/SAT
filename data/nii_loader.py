import pandas as pd
# from typing import List
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import torch
import monai
from einops import repeat, rearrange, reduce
# from skimage import io
import os
import SimpleITK as sitk
from pathlib import Path
import argparse
from tqdm import tqdm
import nibabel as nib
import shutil

class Loader_Wrapper():
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    
    def ACDC(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # merge labels
        mc_mask = [torch.where(mask==3, 1.0, 0.0), torch.where(mask==1, 1.0, 0.0), torch.where(mask==2, 1.0, 0.0)]
        ventricle_cavity = mc_mask[0] + mc_mask[1] # 'left heart ventricle' + 'right heart ventricle'
        mc_mask.append(ventricle_cavity)
        mask = torch.concat(mc_mask, dim=0) # [1, H, W, D] --> [C, H, W, D]
        mask = torch.where(mask>0.5, 1.0, 0.0)
        labels = datum['label'][:3] + ['heart ventricle']
        
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def CHAOS_CT(self, datum:dict) -> tuple:
        """
        liver
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, datum['label'], datum['modality'], datum['image'], datum['mask']
    
    def CHAOS_MRI(self, datum:dict) -> tuple:
        """
        'liver', 
        'right kidney', 
        'left kidney', 
        'spleen'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:4]
        
        # NOTE: merge label
        kidney = mask[1] + mask[2]
        mask = torch.cat((mask, kidney.unsqueeze(0)), dim=0)
        labels.append("kidney")
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, datum['label'], datum['modality'], datum['image'], datum['mask']
    
    def AbdomenCT1K(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:4]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MRSpineSeg(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="ASR", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        mc_masks = []
        labels = datum['label'][:19]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
            
        # merge new labels
        lumbar = torch.zeros_like(mc_masks[0])
        for i in range(1, 6):   # lumbar vertebrae 5 (L5) ~ lumbar vertebrae 5 (L1)
            lumbar += mc_masks[i]
        mc_masks.append(lumbar)
        labels.append('lumbar vertebrae')
            
        thoracic = torch.zeros_like(mc_masks[0])
        for i in range(6, 10):  # thoracic vertebrae 12 (T12) ~ thoracic vertebrae 12 (T9)
            thoracic += mc_masks[i]
        mc_masks.append(thoracic)
        labels.append('thoracic vertebrae')
            
        intervertebral = torch.zeros_like(mc_masks[0])
        for i in range(10, 19): # intervertebral discs between xxx and xxx
            intervertebral += mc_masks[i]
        mc_masks.append(intervertebral)
        labels.append('intervertebral discs')
        
        vertebrae = torch.zeros_like(mc_masks[0])
        for i in range(0, 10):
            vertebrae += mc_masks[i]
        mc_masks.append(vertebrae)
        labels.append('vertebrae')
 
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Liver(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        # 0 is liver, 1 is liver tumor, should be included in liver
        mask[0] += mask[1]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Prostate(self, datum:dict) -> tuple:
        mod2channel = {"T2":0, "ADC":1}
        tmp = datum['image'].split('/')
        mod = tmp[-1]
        channel = mod2channel[mod]
        img_path = '/'.join(tmp[:-1])
        
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
                #monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':img_path, 'label':datum['mask']})
        img = dictionary['image'][channel, :, :, :] # [H, W, D]
        mask = dictionary['label'] # [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        mc_masks.append(mc_masks[0]+mc_masks[1]) 
        labels.append('prostate')
        
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        
        img = repeat(img, 'h w d -> c h w d', c=1)  # [C, H, W, D]
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Pancreas(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        mask[0] += mask[1]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_HepaticVessel(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Spleen(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def WORD(self, datum:dict) -> tuple:
        """
        labels = [
            "liver",
            "spleen",
            "left kidney",  # 3
            "right kidney", # 4
            "stomach",
            "gallbladder",
            "esophagus",
            "pancreas", # 8
            "duodenum",
            "colon",    # 10
            "intestine",
            "adrenal",  # 12
            "rectum",
            "urinary bladder",
            "head of left femur", # 15
            "head of right femur" # 16
            "kidney" = "left kidney"+"right kidney"
            "head of femur" = "head of left femur"+"head of right femur" # 18
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:16]
        
        mc_masks = []
        for i, label in enumerate(datum['label'][:16]):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
            
        # merge label
        mc_masks.append(mc_masks[2]+mc_masks[3])
        labels.append("kidney")
        
        mc_masks.append(mc_masks[14]+mc_masks[15])
        labels.append("head of femur")  
          
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def FLARE22(self, datum:dict) -> tuple:
        """
        'Liver',    # 1
        'Right kidney', # 2
        'Spleen',
        'Pancreas',
        'Aorta',
        'Inferior Vena Cava',
        'Right Adrenal Gland', # 7
        'Left Adrenal Gland', # 8
        'Gallbladder',
        'Esophagus',
        'Stomach',
        'Duodenum',
        'Left kidney'   # 13
        'Kidney' = 'Left kidney' + 'Right kidney'
        'Adrenal Gland' = 'Right Adrenal Gland' + 'Left Adrenal Gland'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:13]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[12])
        labels.append("Kidney")
        mc_masks.append(mc_masks[6]+mc_masks[7])
        labels.append("Adrenal Gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def Couinaud_Liver(self, datum:dict) -> tuple:
        """
        'caudate lobe',     0
        'left lateral superior segment of liver',   1
        'Left lateral inferior segment of liver',   2
        'left medial segment of liver', 3
        'right anterior inferior segment of liver', 4
        'right posterior inferior segment of liver',    5
        'right posterior superior segment of liver',    6
        'right anterior superior segment of liver'  7
        'left lobe of liver' = 1 + 2 + 3
        'right lobe of liver' = 4 + 5 + 6 + 7
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:8]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[1]+mc_masks[2]+mc_masks[3])
        labels.append('left lobe of liver')
        
        mc_masks.append(mc_masks[4]+mc_masks[5]+mc_masks[6]+mc_masks[7])
        labels.append('right lobe of liver')
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2]+mc_masks[3]+mc_masks[4]+mc_masks[5]+mc_masks[6]+mc_masks[7])
        labels.append('liver')
        
        mask = torch.cat(mc_masks, dim=0) # [11, H, W, D]

        mask = (mask-mask.min())/(mask.max()-mask.min()+1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_CT(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("kidney")
        mc_masks.append(mc_masks[10]+mc_masks[11])
        labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_MRI(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("kidney")
        mc_masks.append(mc_masks[10]+mc_masks[11])
        labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def BTCV(self, datum:dict) -> tuple:
        """
        labels = [
            "spleen",
            "right kidney",
            "left kidney",
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "inferior vena cava",
            "portal vein and splenic vein",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:13]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("kidney")
        mc_masks.append(mc_masks[11]+mc_masks[12])
        labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def PARSE2022(self, datum:dict) -> tuple:
        """
        labels = [
            "pulmonary artery",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def SegTHOR(self, datum:dict) -> tuple:
        """
        labels = [
            "esophagus",
            "heart",
            "trachea",
            "aorta",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:4]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MM_WHS_CT(self, datum:dict) -> tuple:
        """
        labels = [
            "myocardium",
            "left heart atrium",
            "left heart ventricle",
            "right heart atrium",
            "right heart ventricle",
            "heart ascending aorta",
            "pulmonary artery",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:7]
        intensity = [205, 420, 500, 550, 600, 820, 850]
        
        mc_masks = []
        for label, value in zip(labels, intensity):
            binary_mask = torch.where(mask==value, 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[3])
        labels.append("heart atrium")
        mc_masks.append(mc_masks[2]+mc_masks[4])
        labels.append("heart ventricle")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MM_WHS_MRI(self, datum:dict) -> tuple:
        """
        labels = [
            "myocardium",
            "left heart atrium",
            "left heart ventricle",
            "right heart atrium",
            "right heart ventricle",
            "heart ascending aorta",
            "pulmonary artery",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:7]
        intensity = [205, 420, 500, 550, 600, 820, 850]
        
        mc_masks = []
        for label, value in zip(labels, intensity):
            binary_mask = torch.where(mask==value, 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[3])
        labels.append("heart atrium")
        mc_masks.append(mc_masks[2]+mc_masks[4])
        labels.append("heart ventricle")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def KiTS23(self, datum:dict) -> tuple:
        """
        labels = [
            "kidney",
            "kidney tumor",
            "kidney cyst",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[0] += mc_masks[1]
        mc_masks[0] += mc_masks[2]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_GLI(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_MEN(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_MET(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_PED(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_SSA(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BTCV_Cervix(self, datum:dict) -> tuple:
        '''
        labels = [
            "urinary bladder",
            "uterus",
            "rectum",
            "small bowel",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="LPS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:4]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def SEGA(self, datum:dict) -> tuple:
        '''
        labels = [
            "aorta",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def Pancreas_CT(self, datum:dict) -> tuple:
        '''
        labels = [
            "pancreas",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def VerSe(self, datum:dict) -> tuple:
        '''
        labels = [
            "cervical vertebrae 1 (c1)",
            "cervical vertebrae 2 (c2)",
            "cervical vertebrae 3 (c3)",
            "cervical vertebrae 4 (c4)",
            "cervical vertebrae 5 (c5)",
            "cervical vertebrae 6 (c6)",
            "cervical vertebrae 7 (c7)", # 6
            "thoracic vertebrae 1 (t1)",
            "thoracic vertebrae 2 (t2)",
            "thoracic vertebrae 3 (t3)",
            "thoracic vertebrae 4 (t4)",
            "thoracic vertebrae 5 (t5)",
            "thoracic vertebrae 6 (t6)",
            "thoracic vertebrae 7 (t7)",
            "thoracic vertebrae 8 (t8)",
            "thoracic vertebrae 9 (t9)",
            "thoracic vertebrae 10 (t10)",
            "thoracic vertebrae 11 (t11)",
            "thoracic vertebrae 12 (t12)", # 18
            "lumbar vertebrae 1 (l1)",
            "lumbar vertebrae 2 (l2)",
            "lumbar vertebrae 3 (l3)",
            "lumbar vertebrae 4 (l4)",
            "lumbar vertebrae 5 (l5)",
            "lumbar vertebrae 6 (l6)", # 24
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="ASR", keys=['image', 'label']),  # IPR
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:26]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        cervical = torch.zeros_like(mc_masks[0])
        for i in range(7):
            cervical += mc_masks[i]
        mc_masks.append(cervical)
        labels.append('cervical vertebrae')

        thoracic = torch.zeros_like(mc_masks[0])
        for i in range(7, 19):
            thoracic += mc_masks[i]
        thoracic += mc_masks[25]
        mc_masks.append(thoracic)
        labels.append('thoracic vertebrae')

        lumbar = torch.zeros_like(mc_masks[0])
        for i in range(19, 25):
            lumbar += mc_masks[i]
        mc_masks.append(lumbar)
        labels.append('lumbar vertebrae')

        vertebrae = torch.zeros_like(mc_masks[0])
        for i in range(26):
            vertebrae += mc_masks[i]
        mc_masks.append(vertebrae)
        labels.append('vertebrae')

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
 
    def LiQA(self, datum:dict) -> tuple:
        """
        labels = [
            'liver',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        # original
        labels = datum['label'][:1]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def ATM22(self, datum:dict) -> tuple:
        """
        labels = [
            'trachea and bronchie',
        ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        # original
        labels = datum['label'][:1]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def IRCADB3D(self, datum:dict) -> tuple:
        """
        labels = [
            'portal vein',
            'bone',
            'liver',
            'liver cyst',
            'liver tumor',
            'artery',
            'biliary system',
            'urinary bladder',
            'gallbladder',
            'heart',
            'kidney',
            'left kidney',
            'left lung',
            'left adrenal gland',
            'left adrenal gland tumor',
            'lung',
            'pancreas',
            'right kidney',
            'right lung',
            'right adrenal gland',
            'right adrenal gland tumor',
            'small bowel',
            'spleen',
            'stomach',
            'adrenal gland',
            'uterus',
            'vena cava',
            'venous system',
        ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [N, H, W, D]
        
        # original
        labels = datum['label']

        # liver cyst / tumor
        if 'liver' in labels:
            organ_idx = labels.index('liver')
            if 'liver cyst' in labels:
                lesion_idx = labels.index('liver cyst')
                mask[organ_idx] += mask[lesion_idx]
            if 'liver tumor' in labels:
                lesion_idx = labels.index('liver tumor')
                mask[organ_idx] += mask[lesion_idx]   
                
        # left adrenal gland tumor
        if 'left adrenal gland' in labels:
            organ_idx = labels.index('left adrenal gland')
            if 'left adrenal gland tumor' in labels:
                lesion_idx = labels.index('left adrenal gland tumor')
                mask[organ_idx] += mask[lesion_idx] 
                
        # right adrenal gland tumor
        if 'right adrenal gland' in labels:
            organ_idx = labels.index('right adrenal gland')
            if 'right adrenal gland tumor' in labels:
                lesion_idx = labels.index('right adrenal gland tumor')
                mask[organ_idx] += mask[lesion_idx]   
                
        # merge left and right
        if 'left adrenal gland' in labels and 'right adrenal gland' in labels and 'adrenal gland' not in labels:
            left_adrenal_gland_idx = labels.index('left adrenal gland')
            right_adrenal_gland_idx = labels.index('right adrenal gland')
            adrenal_gland_mask = torch.zeros_like(img)  
            adrenal_gland_mask[0, :, :, :] += mask[left_adrenal_gland_idx, :, :, :]
            adrenal_gland_mask[0, :, :, :] += mask[right_adrenal_gland_idx, :, :, :]
            mask = torch.concat([mask, adrenal_gland_mask], dim=0)
            labels.append('adrenal gland')
            
        if 'left adrenal gland tumor' in labels and 'right adrenal gland tumor' in labels and 'adrenal gland tumor' not in labels:
            left_adrenal_gland_idx = labels.index('left adrenal gland tumor')
            right_adrenal_gland_idx = labels.index('right adrenal gland tumor')
            adrenal_gland_mask = torch.zeros_like(img)  
            adrenal_gland_mask[0, :, :, :] += mask[left_adrenal_gland_idx, :, :, :]
            adrenal_gland_mask[0, :, :, :] += mask[right_adrenal_gland_idx, :, :, :]
            mask = torch.concat([mask, adrenal_gland_mask], dim=0)
            labels.append('adrenal gland tumor')
            
        if 'left lung' in labels and 'right lung' in labels and 'lung' not in labels:
            left_lung_idx = labels.index('left lung')
            right_lung_idx = labels.index('right lung')
            lung_mask = torch.zeros_like(img)  
            lung_mask[0, :, :, :] += mask[left_lung_idx, :, :, :]
            lung_mask[0, :, :, :] += mask[right_lung_idx, :, :, :]
            mask = torch.concat([mask, lung_mask], dim=0)
            labels.append('lung')
            
        if 'left kidney' in labels and 'right kidney' in labels and 'kidney' not in labels:
            left_kidney_idx = labels.index('left kidney')
            right_kidney_idx = labels.index('right kidney')
            kidney_mask = torch.zeros_like(img)  
            kidney_mask[0, :, :, :] += mask[left_kidney_idx, :, :, :]
            kidney_mask[0, :, :, :] += mask[right_kidney_idx, :, :, :]
            mask = torch.concat([mask, kidney_mask], dim=0)
            labels.append('kidney')
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
  
def Normalization(torch_image, modality):
    if modality.lower() == 'ct':
        lower_bound, upper_bound = -500, 1000
        torch_image = torch.clamp(torch_image, min=lower_bound, max=upper_bound)
        torch_image = (torch_image - torch_image.mean()) / torch_image.std()
    else:
        percentile_00_5, percentile_99_5 = torch.quantile(torch_image, 0.005), torch.quantile(torch_image, 0.995)
        torch_image = torch.clamp(torch_image, min=percentile_00_5.item(), max=percentile_99_5.item())
        torch_image = (torch_image - torch_image.mean()) / torch_image.std()
    return torch_image

def checksample(path2jsonl, sample_idx=0):
    import cv2
    
    """
    path2jsonl: path to the jsonl file
    sample_idx: choose a sample from the jsonl file
    """

    root = '/mnt/hwfile/medai/zhaoziheng/SAM'
    
    loader = Loader_Wrapper()
    
    # 模拟读取
    with open(path2jsonl, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    if sample_idx == -1:
        sample_idx_ls = [i for i in range(0, len(data))]
    else:
        sample_idx_ls = [sample_idx]
    
    for sample_idx in sample_idx_ls:
    
        func_name = data[sample_idx]['dataset']
        batch = getattr(loader, func_name)(data[sample_idx])
        img_tensor, mc_mask, text_ls, modality, image_path, mask_path = batch
        if mc_mask.dtype == torch.bool:
            mc_mask = mc_mask.to(torch.uint8)
        
        """   
        print(mc_mask.shape)
        for i in range(12):
            print(torch.sum(torch.where(mc_mask[i]==1, 1.0, 0.0))+torch.sum(torch.where(mc_mask[i]==0, 1.0, 0.0)))
        exit()
        """
        
        # 检查数据
        dataset_name = data[sample_idx]['dataset']
        assert torch.sum(torch.where(mc_mask==0, 1, 0)).item() + torch.sum(torch.where(mc_mask==1, 1, 0)).item() == mc_mask.shape[0]*mc_mask.shape[1]*mc_mask.shape[2]*mc_mask.shape[3]
        print('* Dataset %s has %d samples *'%(dataset_name, len(data)))
        print('* image path * : ', image_path)
        print('* mask path * : ', mask_path)
        print('* modality * : ', modality)
        print('* labels * : ', text_ls)
        print('* img_tensor.shape * : ', img_tensor.shape)  # [c h w d]
        print('* img_tensor.dtype * : ', img_tensor.dtype)
        print('* mc_mask.shape * : ', mc_mask.shape)    # [c h w d]
        print('* mc_mask.dtype * : ', mc_mask.dtype)
        print('* sum(mc_mask) * : ', torch.sum(mc_mask))
        
        mc_mask = mc_mask.numpy()
        img_tensor = img_tensor.numpy()
        if mc_mask.shape[-1] > 0:
            # 3D按nifiti存
            results = np.zeros((img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])) # hwd
            for j, label in enumerate(text_ls):
                results += mc_mask[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
                Path(f'{root}/visualization/3D/{dataset_name}/(loader_v4)sample_{sample_idx}/segmentations').mkdir(exist_ok=True, parents=True)
                # 每个label单独一个nii.gz
                segobj = nib.nifti2.Nifti1Image(mc_mask[j, :, :, :], np.eye(4))
                nib.save(segobj, f'{root}/visualization/3D/{dataset_name}/(loader_v4)sample_{sample_idx}/segmentations/{label}.nii.gz')
            segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
            nib.save(segobj, f'{root}/visualization/3D/{dataset_name}/(loader_v4)sample_{sample_idx}/seg.nii.gz')
            
            imgobj = nib.nifti2.Nifti1Image(img_tensor[0], np.eye(4))   # hwd
            nib.save(imgobj, f'{root}/visualization/3D/{dataset_name}/(loader_v4)sample_{sample_idx}/img.nii.gz')

        # 按slice存
        for slice_idx in tqdm(range(mc_mask.shape[-1])):
            Path(f'{root}/visualization/3D/%s/(loader_v4)sample_%d/slice_%d'%(dataset_name, sample_idx, slice_idx)).mkdir(parents=True, exist_ok=True)
            Path(f'{root}/visualization/3D/%s/(loader_v4)sample_%d/image_series'%(dataset_name, sample_idx)).mkdir(parents=True, exist_ok=True)
            img = rearrange(img_tensor[:, :, :, slice_idx], 'c h w -> h w c') # [H, W, C]
            cv2.imwrite(f'{root}/visualization/3D/%s/(loader_v4)sample_%d/slice_%d/img.jpg'%(dataset_name, sample_idx, slice_idx), img*255.0)
            cv2.imwrite(f'{root}/visualization/3D/%s/(loader_v4)sample_%d/image_series/slice_%d.jpg'%(dataset_name, sample_idx, slice_idx), img*255.0)
            for label_idx, text in tqdm(enumerate(text_ls)):
                msk = mc_mask[label_idx, :, :, slice_idx]
                if np.sum(msk) > 0:
                    """
                    # the bbox
                    non_zero_coordinates = np.nonzero(msk) # ([...], [...])
                    y1, x1 = np.min(non_zero_coordinates[0]).item(), np.min(non_zero_coordinates[1]).item()
                    y2, x2 = np.max(non_zero_coordinates[0]).item(), np.max(non_zero_coordinates[1]).item()
                    print('slice no.%d, label no.%d : %s, [x1, y1, x2, y2] : [%d, %d, %d, %d]'%(slice_idx, label_idx, text, x1, y1, x2, y2))
                    """
                    print('slice no.%d, label no.%d : %s'%(slice_idx, label_idx, text))
                    cv2.imwrite(f'{root}/visualization/3D/%s/(loader_v4)sample_%d/slice_%d/%d_%s_msk.jpg'%(dataset_name, sample_idx,  slice_idx, label_idx, text), msk*255.0)
                    if img.shape[2] == 1:
                        img = repeat(img, 'h w c -> h w (c r)', r=3)
                    overlap = repeat(msk, 'h w -> h w c', c=3) # colorful mask H, W, C
                    img = np.float32(img)
                    overlap = np.float32(overlap)
                    overlap = cv2.add(img*255.0, overlap*255.0)
                    cv2.imwrite(f'{root}/visualization/3D/%s/(loader_v4)sample_%d/slice_%d/%d_%s_seg.jpg'%(dataset_name, sample_idx,  slice_idx, label_idx, text), overlap)
        
        shutil.copy(path2jsonl, f'{root}/visualization/3D/%s/(loader_v4)sample_%d/source_file.jsonl'%(dataset_name, sample_idx))

def checkdataset(path2jsonl):
    """
    path2jsonl: path to the jsonl file
    sample_idx: choose a sample from the jsonl file
    """
    import traceback

    root = '/mnt/hwfile/medai/zhaoziheng/SAM'

    loader = Loader_Wrapper()
    
    # 模拟读取
    with open(path2jsonl, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    dataset_with_error = set()

    for sample in tqdm(data, desc=f'checking each sample ... ...'):
        func_name = sample['dataset']
        try:
            batch = getattr(loader, func_name)(sample)
            img_tensor, mc_mask, text_ls, modality, image_path, mask_path = batch
            if mc_mask.dtype == torch.bool:
                mc_mask = mc_mask.to(torch.uint8)
            assert torch.sum(torch.where(mc_mask==0, 1, 0)).item() + torch.sum(torch.where(mc_mask==1, 1, 0)).item() == mc_mask.shape[0]*mc_mask.shape[1]*mc_mask.shape[2]*mc_mask.shape[3]
            assert img_tensor.shape[1] == mc_mask.shape[1] and img_tensor.shape[2] == mc_mask.shape[2] and img_tensor.shape[3] == mc_mask.shape[3], f'image {img_tensor.shape} != mask {mc_mask.shape} in {sample["image"]}'
            assert mc_mask.shape[0] == len(text_ls), f'mask {mc_mask.shape} != {len(text_ls)} labels in {sample["image"]}'
        except:
            if sample["dataset"] not in dataset_with_error:
                print(f'Meet Error in {sample["dataset"]}')
                dataset_with_error.add(sample["dataset"])
            
            info = traceback.format_exc()
            Path(f'{root}/visualization/3D/{sample["dataset"]}').mkdir(exist_ok=True, parents=True)
            with open(f'{root}/visualization/3D/{sample["dataset"]}/(loader_v4)load_error.text', 'w') as f:
                f.write(f'** {sample["dataset"]} ** {sample["patient_id"]} **\n')
                f.write(info)
                f.write('\n')
                f.write('\n')
                
if __name__ == '__main__':       
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2jsonl', type=str)
    parser.add_argument('--i', type=int, help='index of sample to check, visualize all if -1')
    config = parser.parse_args()

    # path2jsonl = 'datasets/%s/%s.jsonl'%(config.dataset_name, config.dataset_name)
    if config.i is not None:
        checksample(config.path2jsonl, config.i)
    else:
        checkdataset(config.path2jsonl)