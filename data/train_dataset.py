import os
import random
import math

from einops import rearrange, repeat, reduce
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import traceback
from tqdm import tqdm
from monai.transforms import (
    Compose,
    RandShiftIntensityd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandScaleIntensityd,
    #RandScaleIntensityFixedMeand,
    #RandSimulateLowResolutiond,
    RandAdjustContrastd
)
import time

from train.dist import is_master

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False         
    
class Med_SAM_Dataset(Dataset):
    def __init__(self, 
                 jsonl_file, 
                 dataset_config,
                 crop_size=[288,288,96], 
                 max_queries=16, 
                 allow_repeat=True):
        """
        Assemble segmentation datasets
        
        Args:
            jsonl_file (str): a jsonl contains all train sample information
            crop_size (int, optional): size of the input cropped image. Defaults to [288,288,96].
            max_queries (int, optional): max number of queries. Defaults to 32.
            dataset_config (str, optional): a path to config file, defining the sampling, loading parameters of each dataset etc
            allow_repeat (bool, optional): sample for multiply times to accelerate convergency. Defaults to True.
        """
        self.crop_size = crop_size
        self.max_queries = max_queries
        
        # load data configs
        with open(dataset_config, 'r') as f:
            self.dataset_config = json.load(f)
        
        # load 
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
        # statistics the size of each dataset and the repeated times and the sampled times within a log interval
        datasets_dist = [l['dataset'] for l in lines]
        self.datasets = set(datasets_dist)
        self.datasets_size = {}
        self.datasets_repeat_times = {}
        for dataset in self.datasets:
            self.datasets_size[dataset] = datasets_dist.count(dataset)
            self.datasets_repeat_times[dataset] = 0       
    
        self.data_3d = []   # list of data samples
        self.sample_weight_3d = []  # and their sampling weight
        count_3d_repeat = 0
        
        for sample in lines:
            
            # sampling weight : inverse to square root of dataset size
            size = self.datasets_size[sample['dataset']]
            weight = 1 / (math.sqrt(size))
            # sampling weight : allow manual adjustment in data config file
            weight = weight * self.dataset_config[sample['dataset']]['sampling_weight']
            
            # repeat times for label num
            query_repeat_times = max(1, (len(sample['label']) / max_queries))
            # repeat times for roi size
            if sample['roi_y1x1z1_y2x2z2']:
                y1, x1, z1, y2, x2, z2 = sample['roi_y1x1z1_y2x2z2']
                h_repeat_times = max(1, ((y2-y1) / crop_size[0]))
                w_repeat_times = max(1, ((x2-x1) / crop_size[1]))
                d_repeat_times = max(1, ((z2-z1) / crop_size[2]))
                size_repeat_times = h_repeat_times * w_repeat_times * d_repeat_times
            else:
                size_repeat_times = 1
                
            # not repeat
            if not allow_repeat:
                size_repeat_times = query_repeat_times = 1
            # allow repeat
            repeat_times = round(size_repeat_times * query_repeat_times)  # e.g. 1.5 * 2.5 = 3.75 --> 4
            if 'is_3D' not in sample or sample['is_3D']=='3D':
                for i in range(round(repeat_times)):
                    self.data_3d.append(sample)
                    self.sample_weight_3d.append(weight)
                count_3d_repeat += (repeat_times - 1)
                self.datasets_repeat_times[sample['dataset']] += (repeat_times - 1)
            elif sample['is_3D']=='2D':
                for i in range(round(repeat_times)):
                    self.data_2d.append(sample)
                    self.sample_weight_2d.append(weight)
                count_2d_repeat += (repeat_times - 1)
                self.datasets_repeat_times[sample['dataset']] += (repeat_times - 1)
            else:
                raise ValueError(f"data type {sample['is_3D']} is neither 2D or 3D")
        
        if is_master():
            print(f'** DATASET ** {len(lines)} unique 3D samples are loaded, {count_3d_repeat} samples are repeated')  
            print(f'** DATASET ** In total {len(self.datasets)} datasets.\n')
            print(f'** DATASET ** Size, Repeated Times and Repeat/Size Ratio for each dataset:\n')
            for k,repeated_times in self.datasets_repeat_times.items():
                size = self.datasets_size[k]
                print(f'{k} : {size}/{repeated_times} = {repeated_times/size}')
        
        # data augmentation (tailor for each dataset)
        self.augmentator = {}
        for dataset in self.datasets:
            config = self.dataset_config[dataset]['augmentation']
            aug_ls = []
            if 'RandZoom' in config:
                aug_ls.append(RandZoomd(
                        keys=["image", "label"], 
                        mode=['area', 'nearest'],
                        min_zoom=config['RandZoom']['min_zoom'],
                        max_zoom=config['RandZoom']['max_zoom'],
                        prob=config['RandZoom']['prob'],
                    )
                )
            if 'RandGaussianNoise' in config:
                aug_ls.append(
                    RandGaussianNoised(
                        keys=['image'],
                        prob=config['RandGaussianNoise']['prob'],
                        mean=config['RandGaussianNoise']['mean'],
                        std=0.1
                    )
                )
            if 'RandGaussianSharpen' in config:
                aug_ls.append(
                    RandGaussianSharpend(
                        keys=['image'],
                        prob=config['RandGaussianSharpen']['prob'],
                    )
                )
            if 'RandScaleIntensity' in config:
                aug_ls.append(
                    RandScaleIntensityd(
                        keys=['image'],
                        factors=config['RandScaleIntensity']['factors'],
                        prob=config['RandScaleIntensity']['prob']
                    )
                )
            """if 'RandScaleIntensityFixedMean' in config:
                aug_ls.append(
                    RandScaleIntensityFixedMeand(
                        keys=['image'],
                        factors=config['RandScaleIntensityFixedMean']['factors'],
                        prob=config['RandScaleIntensityFixedMean']['prob']
                    )
                )
            if 'RandSimulateLowResolution' in config:
                aug_ls.append(
                    RandSimulateLowResolutiond(
                        keys=['image'],
                        prob=config['RandSimulateLowResolution']['prob']
                    )
                )"""
            if 'RandAdjustContrastInvert' in config:
                aug_ls.append(
                    RandAdjustContrastd(
                        keys=['image'],
                        #retain_stats=config['RandAdjustContrastInvert']['retain_stats'],
                        #invert_image=config['RandAdjustContrastInvert']['invert_image'],
                        gamma=config['RandAdjustContrastInvert']['gamma'],
                        prob=config['RandAdjustContrastInvert']['prob']
                    )
                )
            if 'RandAdjustContrast' in config:
                aug_ls.append(
                    RandAdjustContrastd(
                        keys=['image'],
                        #retain_stats=config['RandAdjustContrast']['retain_stats'],
                        #invert_image=config['RandAdjustContrast']['invert_image'],
                        gamma=config['RandAdjustContrast']['gamma'],
                        prob=config['RandAdjustContrast']['prob']
                    )
                )
            if len(aug_ls) > 0:
                self.augmentator[dataset] = Compose(aug_ls)
        
    def __len__(self):
        return 1000000000 # life long training ... (10e9)
    
    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        if contains(mod, ['us', 'ultrasound']):
            return 'us'
        else:
            return mod
    
    def _pad_if_necessary(self, image=None, mask=None):
        # image size >= crop size 
        if not (image is None):
            c, h, w, d = image.shape
            croph, cropw, cropd = self.crop_size
            pad_in_h = 0 if h >= croph else croph - h
            pad_in_w = 0 if w >= cropw else cropw - w
            pad_in_d = 0 if d >= cropd else cropd - d
            if pad_in_h + pad_in_w + pad_in_d > 0:
                pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
                image = F.pad(image, pad, 'constant', 0)   # chwd
        
        if not (mask is None):
            n, h, w, d = mask.shape
            croph, cropw, cropd = self.crop_size
            pad_in_h = 0 if h >= croph else croph - h
            pad_in_w = 0 if w >= cropw else cropw - w
            pad_in_d = 0 if d >= cropd else cropd - d
            if pad_in_h + pad_in_w + pad_in_d > 0:
                pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
                mask = F.pad(mask, pad, 'constant', 0)   # nhwd
        
        return image, mask
    
    def _crop(self, image, datum, roi_crop_prob, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        if (imgh - croph) > 0 or (imgw - cropw) > 0 or (imgd - cropd) > 0:
            # need crop
            if not (True in datum['renorm_y1x1z1_y2x2z2']) or random.random() > roi_crop_prob:
                # no roi region
                image, y1x1z1_y2x2z2 = self._random_crop(image)
            else:
                # 100% roi crop
                image, y1x1z1_y2x2z2 = self._roi_crop(image, datum, label_based_crop_prob, uncenter_prob)
        else:
            y1x1z1_y2x2z2 = [0, 0, 0, imgh, imgw, imgd]
                
        return image, y1x1z1_y2x2z2
    
    def _roi_crop(self, image, datum, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        
        if random.random() < label_based_crop_prob:
            # find a pos label and crop based on it (ensure at least one pos label before roi crop
            pos_label_idx_ls = [i for i, t_or_f in enumerate(datum['renorm_y1x1z1_y2x2z2']) if t_or_f]
            pos_label_idx = random.sample(pos_label_idx_ls, 1)[0]
            mask_to_select = self._load_mask(datum, [datum['label'][pos_label_idx]], [datum['renorm_y1x1z1_y2x2z2'][pos_label_idx]])    # 1 h w d
            mask_to_select = mask_to_select[0, :, :, :]  # h w d 
        else:
            # crop based on all labels
            _, h, w, d = datum['chwd']
            mask_to_select = torch.zeros((h, w, d), dtype=torch.bool)
            y1, x1, z1, y2, x2, z2 = datum['roi_y1x1z1_y2x2z2']
            mask_to_select[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(f"{datum['renorm_segmentation_dir']}.npy"))
        
        # select a voxel
        voxels_foreground = torch.nonzero(mask_to_select, as_tuple=True)   # (tensor(...), tensor(...), tensor(...))
        selected_index = random.randint(0, voxels_foreground[0].shape[0]-1)
        selected_voxel = (voxels_foreground[0][selected_index].item(), voxels_foreground[1][selected_index].item(), voxels_foreground[2][selected_index].item())
        
        # check the boundary
        if selected_voxel[0] - croph // 2 > 0:
            start_y = selected_voxel[0] - croph // 2
            if start_y + croph < imgh:
                end_y = start_y + croph
            else:
                end_y = imgh
                start_y = imgh-croph
        else:
            start_y = 0
            end_y = croph
            
        if selected_voxel[1] - cropw // 2 > 0:
            start_x = selected_voxel[1] - cropw // 2
            if start_x + cropw < imgw:
                end_x = start_x + cropw
            else:
                end_x = imgw
                start_x = imgw-cropw
        else:
            start_x = 0
            end_x = cropw

        if selected_voxel[2] - cropd // 2 > 0:
            start_z = selected_voxel[2] - cropd // 2
            if start_z + cropd < imgd:
                end_z = start_z + cropd
            else:
                end_z = imgd
                start_z = imgd-cropd
        else:
            start_z = 0
            end_z = cropd  
        
        # randomly shift the crop (must contain the selected voxel
        if random.random() < uncenter_prob:
            y_left_space = min(start_y - 0, end_y - selected_voxel[0])
            y_right_space = min(imgh - end_y, selected_voxel[0] - start_y)
            y_adjust = random.randint(-1 * y_left_space, y_right_space)
            start_y += y_adjust
            end_y += y_adjust
            
            x_left_space  = min(start_x-0, end_x-selected_voxel[1])
            x_right_space = min(imgw-end_x, selected_voxel[1]-start_x)
            x_adjust = random.randint(-1*x_left_space, x_right_space)
            start_x += x_adjust
            end_x += x_adjust

            z_left_space = min(start_z - 0, end_z - selected_voxel[2])
            z_right_space = min(imgd - end_z, selected_voxel[2] - start_z)
            z_adjust = random.randint(-1 * z_left_space, z_right_space)
            start_z += z_adjust
            end_z += z_adjust
            
        # crop
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]

        return crop_image, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _random_crop(self, image):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        # 
        start_y = random.randint(0, imgh - croph)
        end_y = start_y + croph
        start_x = random.randint(0, imgw - cropw)
        end_x = start_x + cropw
        start_z = random.randint(0, imgd - cropd)
        end_z = start_z + cropd
        #
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]
        
        return crop_image, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _select_pos_labels(self, label_index_ls, is_pos_ls, neg_label_ratio_threshold):
        """
        尽可能多采positive的label同时控制negative的数量不能超过positive的一定比例
        
        Args:
            label_index_ls (List of int) : candidate labels (channel index in segmentation mask)
            is_pos_ls (List of bool) : positive label (True) or not (False), equal length to label_index_ls
        
        Returns:
            chosen_label_index_ls (List of int) : chosen subset of label_index_ls
            chosen_is_pos (List of bool) : chosen subset of is_pos_ls
        """
        # divide all the labels into pos and neg
        pos_label_index_ls = []
        neg_label_index_ls = []
        for i, is_pos in zip(label_index_ls, is_pos_ls):
            if is_pos:
                pos_label_index_ls.append(i)
            else:
                neg_label_index_ls.append(i)
        pos_num = len(pos_label_index_ls)
        neg_num = len(neg_label_index_ls)
        
        if pos_num == 0:
            # degrad to random sample
            sample_num = min(self.max_queries, len(label_index_ls))
            chosen_label_index_ls = random.sample(label_index_ls, sample_num)
            chosen_is_pos = [False] * sample_num
            return chosen_label_index_ls, chosen_is_pos
        
        # indicate each sample is pos or neg
        chosen_is_pos = []
        
        if pos_num <= self.max_queries:
            # all pos labels are included, then sample some neg labels
            chosen_label_index_ls = pos_label_index_ls 
            chosen_is_pos += [True] * pos_num
            max_neg_num = int(neg_label_ratio_threshold * pos_num)    # neg label num < (pos label num) * x%
            left_pos_num = min(self.max_queries-pos_num, max_neg_num)   # neg label num < self.max_queries-pos_num
            if neg_num <= left_pos_num:
                # neg are all sampled
                chosen_label_index_ls += neg_label_index_ls
                chosen_is_pos += [False] * neg_num
            else:
                # neg are sampled to control the ratio and max label num
                chosen_label_index_ls += random.sample(neg_label_index_ls, left_pos_num)
                chosen_is_pos += [False] * left_pos_num
        else:
            # no neg labels are sampled
            chosen_label_index_ls = random.sample(pos_label_index_ls, self.max_queries)
            chosen_is_pos += [True] * self.max_queries

        return chosen_label_index_ls, chosen_is_pos
    
    def _load_mask(self, datum, labels_to_load, y1x1z1_y2x2z2_to_load):
        """
        加载segmentation mask
        Args:
            datum (dict): sample info (a line from jsonl file

        Returns:
            mc_mask: (N, h, w, d)
            labels: list of N str
            is_pos: lits of True/False
        """
        _, h, w, d = datum['chwd']
        mask_paths = [f"{datum['renorm_segmentation_dir']}/{label}.npy" for label in labels_to_load] # /remote-home/share/SAM/processed_files/MSD_Liver/segmentation/27/liver tumor.npy
        y1x1z1_y2x2z2_ls = y1x1z1_y2x2z2_to_load
        
        mc_mask = []
        is_pos = []
        for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
            mask = torch.zeros((h, w, d), dtype=torch.bool)
            # not empty, load and embed non-empty cropped_volume
            if y1x1z1_y2x2z2 != False:
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
                is_pos.append(True)
            else:
                is_pos.append(False)
            mc_mask.append(mask)
            
        mc_mask = torch.stack(mc_mask, dim=0)   # n h w d
        
        return mc_mask
    
    def _load_image(self, datum):
        path = datum['renorm_image']
        image = torch.tensor(np.load(path))

        return image # chwd
    
    def is_overlap(self, a_y1x1z1_y2x2z2, b_y1x1z1_y2x2z2):
        # judge is overlap or not between two cubes
        a_y1, a_x1, a_z1, a_y2, a_x2, a_z2 = a_y1x1z1_y2x2z2
        b_y1, b_x1, b_z1, b_y2, b_x2, b_z2 = b_y1x1z1_y2x2z2
        overlap_x = not (a_x2 < b_x1 or b_x2 < a_x1)
        overlap_y = not (a_y2 < b_y1 or b_y2 < a_y1)
        overlap_z = not (a_z2 < b_z1 or b_z2 < a_z1)
        return overlap_x and overlap_y and overlap_z
    
    def _find_pos_labels_in_crop(self, crop_y1x1z1_y2x2z2, labels_y1x1z1_y2x2z2):
        is_pos = []
        for y1x1z1_y2x2z2 in labels_y1x1z1_y2x2z2:
            if y1x1z1_y2x2z2 and self.is_overlap(y1x1z1_y2x2z2, crop_y1x1z1_y2x2z2):
                is_pos.append(True)
            else:
                is_pos.append(False)
        return is_pos
    
    def get_size_and_repeat(self, dataset_name):
        return self.datasets_size[dataset_name], self.datasets_repeat_times[dataset_name]
        
    def __getitem__(self, idx):
        while True:
            try: 
                sample = random.choices(self.data_3d, weights=self.sample_weight_3d)[0]
                
                # load image
                image = self._load_image(sample)
                
                # # simple check
                # _, H, W, D = image.shape
                # N, mH, mW, mD = mask.shape
                # assert H == mH and W == mW and D == mD, f'image shape {H, W, D} inconsistent with mask shape {mH, mW, mD}'
                # assert N == len(labels_to_load) == len(is_pos_in_volume), f'query num {len(labels_to_load)} inconsistent with gt mask channels {N}'
                
                # merge modality, e.g. t1 -> mri
                modality = sample['modality']
                modality = self._merge_modality(modality.lower())                
                
                # pad image
                image, _ = self._pad_if_necessary(image, mask=None)
                
                # crop image
                roi_crop_prob = self.dataset_config[sample['dataset']]['foreground_crop_prob']
                label_based_crop_prob = self.dataset_config[sample['dataset']]['label_based_crop_prob']
                uncenter_prob = self.dataset_config[sample['dataset']]['uncenter_prob']
                image, y1x1z1_y2x2z2 = self._crop(image, sample, roi_crop_prob, label_based_crop_prob, uncenter_prob)

                # for all the label in this sample, check if positive in the cropped patch (a quick check via cube overlaps)
                is_pos_in_crop = self._find_pos_labels_in_crop(y1x1z1_y2x2z2, sample['renorm_y1x1z1_y2x2z2'])   # [label1, label2, ....] --> [True, False, ...]
                
                # sample from all the labels based on the cropped patch (to balance pos and neg labels)
                pos_label_first_prob = self.dataset_config[sample['dataset']]['pos_label_first_prob']
                neg_label_ratio_threshold = self.dataset_config[sample['dataset']]['neg_label_ratio_threshold']
                all_label_index_ls = [i for i in range(len(is_pos_in_crop))]
                pos_first = random.random() < pos_label_first_prob
                if pos_first:
                    # sample pos labels as many as possible (could be false pos?
                    # chosen_label_index_ls: index of the chosen labels;
                    # is_pos_ls : True/False;
                    chosen_label_index_ls, is_pos_ls = self._select_pos_labels(all_label_index_ls, is_pos_in_crop, neg_label_ratio_threshold)   # [label1, label2, ....], [True, False, ...]
                else:
                    chosen_label_index_ls = random.sample(all_label_index_ls, min(self.max_queries, len(all_label_index_ls)))
                
                # so these are the chosen labels  
                chosen_label = [sample['label'][i] for i in chosen_label_index_ls]
                chosen_y1x1z1_y2x2z2 = [sample['renorm_y1x1z1_y2x2z2'][i] for i in chosen_label_index_ls]
                
                # load their masks
                mask = self._load_mask(sample, chosen_label, chosen_y1x1z1_y2x2z2)
                
                # pad and crop mask (according to the cropped patch)
                _, mask = self._pad_if_necessary(image=None, mask=mask)
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                mask = mask[:, y1:y2, x1:x2, z1:z2]
                
                # if positive first, then strictly control positive ratio
                # filter false positive labels (the cube of the annotation overlap with the cropped patch, but still negative)
                if pos_first:
                    filtered_indexes = []
                    for i, is_pos in enumerate(is_pos_ls):
                        if (not is_pos) or (not torch.all(mask[i])):
                            filtered_indexes.append(i)
                    mask = mask[filtered_indexes, :, :, :]  
                    chosen_label = [chosen_label[i] for i in filtered_indexes]    
                    is_pos_ls = [is_pos_ls[i] for i in filtered_indexes]
                
                # regular augmentation if needed
                if sample['dataset'] in self.augmentator:
                    data_dict = {'image': image, 'label': mask}
                    aug_data_dict = self.augmentator[sample['dataset']](data_dict)
                    image, mask = aug_data_dict['image'], aug_data_dict['label']
                    
                break
            except SystemExit:
                exit()
            except:
                # record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {sample["dataset"]} *** {sample["image"]} ***\n')
                print(traceback_info)

        return {'image':image, 'mask':mask, 'text':chosen_label, 'modality':modality, 'image_path':sample['renorm_image'], 'mask_path':sample['renorm_segmentation_dir'], 'dataset':sample['dataset'], 'y1x1z1_y2x2z2':y1x1z1_y2x2z2}