import os
import random
import math
import warnings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib
from einops import rearrange, repeat, reduce
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import traceback
from tqdm import tqdm
import time

from train.dist import is_master
from data.augmentation import get_SAT_augmentator, get_nnUNet_augmentator
from data.data_loader_cvpr2025challenge import NAME2LOADER

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
                 text_prompts_json,
                 dataset_config,
                 crop_size=[288,288,96], 
                 max_queries=16, 
                 allow_repeat=True,
                 nnUNet_aug=True):
        """
        Assemble segmentation datasets
        
        Args:
            jsonl_file (_type_): a jsonl contains all train sample information
            crop_size (int, optional): _description_. Defaults to [288,288,96].
            max_queries (int, optional): _description_. Defaults to 32.
            dataset_config (str, optional): a path to config file, defining the sampling, loading parameters of each dataset etc
            allow_repeat (bool, optional): sample for multiply times to accelerate convergency. Defaults to True.
        """
        
        self.crop_size = crop_size
        self.max_queries = max_queries
        
        # load data configs
        with open(dataset_config, 'r') as f:
            self.dataset_config = json.load(f)
        
        # load samples
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
        # load intensity 2 label json
        with open(text_prompts_json, 'r') as f:
            self.text_prompts_json = json.load(f)
        
        # statistics the size of each dataset and the repeated times and the sampled times within a log interval
        datasets_dist = [l['dataset'] for l in lines]
        self.datasets = set(datasets_dist)
        self.datasets_size = {}
        self.datasets_repeat_times = {}
        for dataset in self.datasets:
            self.datasets_size[dataset] = datasets_dist.count(dataset)
            self.datasets_repeat_times[dataset] = 0       
    
        self.data = []   # list of data samples
        self.sample_weight = []  # and their sampling weight
        count_repeat = 0
        
        for sample in lines:
            
            # sampling weight : inverse to square root of dataset size
            size = self.datasets_size[sample['dataset']]
            weight = 1 / (math.sqrt(size))
            # sampling weight : allow manual adjustment in data config file
            weight = weight * self.dataset_config[sample['dataset']]['sampling_weight']
            
            # repeat times for label num
            label_num = len(self.text_prompts_json[sample['dataset']]) - 1 # exclude instance label
            query_repeat_times = max(1, (label_num / max_queries))
            # repeat times for roi size
            if 'roi_y1x1z1_y2x2z2' in sample and sample['roi_y1x1z1_y2x2z2']:
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
            for i in range(round(repeat_times)):
                self.data.append(sample)
                self.sample_weight.append(weight)
            count_repeat += (repeat_times - 1)
            self.datasets_repeat_times[sample['dataset']] += (repeat_times - 1)
        self.cumulative_weights = np.cumsum(self.sample_weight)
            
        """
        # determine sample weight and num
        self.num_2d = 0
        self.num = len(self.data)
        self.data_split = {'2d':[0, self.num_2d], '3d':[self.num_2d, self.num_2d+self.num]}
        """
        
        if is_master():
            print(f'** DATASET ** {len(lines)} unique 3D samples are loaded, {count_repeat} samples are repeated')  
            print(f'** DATASET ** In total {len(self.datasets)} datasets.\n')
            print(f'** DATASET ** Size, Repeated Times and Repeat/Size Ratio for each dataset:\n')
            for k,repeated_times in self.datasets_repeat_times.items():
                size = self.datasets_size[k]
                print(f'{k} : {size}/{repeated_times} = {repeated_times/size}')
        
        # data augmentation (tailor for each dataset)
        self.nnUNet_aug = nnUNet_aug
        if nnUNet_aug:
            self.augmentator = get_nnUNet_augmentator(self.datasets, self.crop_size[0])
        else:
            self.augmentator = get_SAT_augmentator(self.dataset_config, self.datasets)
        
    def __len__(self):
        # DEBUG
        # return len(self.data)
        return 1000000000 # life long training ... (10e9)

    def _merge_modality(self, mod):
        if contains(mod, ['mr', 't1', 't2', 'mri', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        if contains(mod, ['us', 'us3d', 'ultrasound']):
            return 'us'
        if contains(mod, ['microscopy']):
            return 'microscopy'
        else:
            raise ValueError(f'Unknown modality {mod}')
    
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
    
    def _crop(self, image, mc_mask, is_roi_crop, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        if (imgh - croph) > 0 or (imgw - cropw) > 0 or (imgd - cropd) > 0:
            # need crop
            if (not mc_mask.any()) or (not is_roi_crop):
                # no roi region
                image, y1x1z1_y2x2z2 = self._random_crop(image)
            else:
                # 100% roi crop
                image, y1x1z1_y2x2z2 = self._roi_crop(image, mc_mask, label_based_crop_prob, uncenter_prob)
        else:
            y1x1z1_y2x2z2 = [0, 0, 0, imgh, imgw, imgd]
                
        return image, y1x1z1_y2x2z2
    
    def _roi_crop(self, image, mc_mask, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        
        if random.random() < label_based_crop_prob:
            # find a pos label and crop based on it (ensure at least one pos label before roi crop
            pos_label_idx_ls = [i for i in range(mc_mask.shape[0]) if mc_mask[i].any()]
            pos_label_idx = random.sample(pos_label_idx_ls, 1)[0]
            mask_to_select = mc_mask[pos_label_idx, :, :, :]  # h w d 
        else:
            # crop based on all labels
            mask_to_select = mc_mask.any(dim=0)
        
        # select a voxel
        voxels_foreground = torch.nonzero(mask_to_select, as_tuple=True)
        selected_index = random.randint(0, len(voxels_foreground[0])-1)
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
        if not crop_image.is_contiguous():
            crop_image = crop_image.contiguous()

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
        if not crop_image.is_contiguous():
            crop_image = crop_image.contiguous()
        
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
    
    def sc_mask_to_mc_mask(self, sc_mask, label_values_ls):
        assert sc_mask.ndim == 3
        h, w, d = sc_mask.shape
        n = len(label_values_ls)
        mc_mask = np.zeros((n, h, w, d), dtype=bool)
        for i, label_value in enumerate(label_values_ls):
            mc_mask[i] = (sc_mask == label_value)
        return mc_mask

    def select_text_prompts(self, lists):
        """
        Select text prompts
        
        Args:
            lists (List): lists of text prompt strings

        Returns:
            selected N elements, and which label they from
        """
        
        # 为每个非空列表生成（原始索引，打乱后的队列）
        queues = []
        for orig_idx, lst in enumerate(lists):
            if lst:
                shuffled = lst.copy()
                random.shuffle(shuffled)  # 初始化打乱顺序
                queues.append((orig_idx, shuffled))  # 保存原始索引和队列
        
        # 随机打乱队列顺序，确保均匀选择列表
        random.shuffle(queues)
        
        collected_elements = []
        source_indices = []
        
        while len(collected_elements) < self.max_queries and queues:
            progress = False
            
            # 倒序遍历避免索引错位
            for i in reversed(range(len(queues))):
                if len(collected_elements) >= self.max_queries:
                    break
                orig_idx, elements = queues[i]
                if elements:
                    # 弹出元素并记录来源索引
                    element = elements.pop()
                    collected_elements.append(element)
                    source_indices.append(orig_idx)
                    progress = True
                    # 如果队列为空，移除它
                    if not elements:
                        queues.pop(i)
            
            if not progress:
                break  # 所有队列已空
        
        return collected_elements[:self.max_queries], source_indices[:self.max_queries]
    
    def load_npz_data(self, dataset_name, data_path):
        data = np.load(data_path)
        sample_name = os.path.basename(data_path)[:-4]
        
        img = data['imgs'].astype(np.float32)  # 0~255
        sc_mask = data['gts'].astype(np.float32)
        spacing = data['spacing'].tolist()
        
        # WARNING Since we dont have nii data, we dont normalize orientation/spacing in the loader any more
        img, sc_mask, _ = NAME2LOADER[dataset_name](sample_name, img, sc_mask, spacing)
        img = img[np.newaxis, :, :, :]  # 1 h w d
        
        label_2_text_prompt = self.text_prompts_json[dataset_name] # '1':['xxx', 'xxx', ...]
        label_values_ls = list(label_2_text_prompt.keys())
        label_values_ls = [int(v) for v in label_values_ls if v!='instance_label']
        text_prompt_ls = list(label_2_text_prompt.values()) # list of list of str
        text_prompt_ls = [ls for ls in text_prompt_ls if isinstance(ls, list)]
        
        mc_mask = self.sc_mask_to_mc_mask(sc_mask, label_values_ls)  # n h w d
        
        return torch.from_numpy(img), torch.from_numpy(mc_mask), text_prompt_ls
        
    def __getitem__(self, idx):
        while True:
            try: 
                # sample = random.choices(self.data, weights=self.sample_weight)[0]
                rand_val = random.random() * self.cumulative_weights[-1]
                sample_idx = np.searchsorted(self.cumulative_weights, rand_val)
                sample = self.data[sample_idx]
                
                dataset_name = sample['dataset']
                data_path = sample['data']

                image, mask, labels = self.load_npz_data(dataset_name, data_path)
                assert mask.dtype == torch.bool
                    
                modality = sample['data'].split('/')[-3]
                modality = self._merge_modality(modality.lower())   

                # pad image
                image, mask = self._pad_if_necessary(image, mask)
                assert mask.dtype == torch.bool
                
                # crop image
                roi_crop_prob = self.dataset_config[dataset_name]['foreground_crop_prob']
                is_roi_crop = random.random() < roi_crop_prob
                label_based_crop_prob = self.dataset_config[dataset_name]['label_based_crop_prob']
                uncenter_prob = self.dataset_config[dataset_name]['uncenter_prob']
                image, y1x1z1_y2x2z2 = self._crop(image, mask, is_roi_crop, label_based_crop_prob, uncenter_prob)
                start_y, start_x, start_z, end_y, end_x, end_z = y1x1z1_y2x2z2
                
                # crop mask
                mask = mask[:, start_y:end_y, start_x:end_x, start_z:end_z]
                if not mask.is_contiguous():
                    mask = mask.contiguous()

                # for all the label in this sample, check if positive in the cropped patch
                # is_pos_in_crop = [mask[i].any().item() for i in range(mask.shape[0])]
                # More efficient batch operation
                mask_any = mask.view(mask.shape[0], -1).any(dim=1)  # Shape: (n_labels,)
                is_pos_in_crop = mask_any.tolist()  # Single GPU-CPU transfer
                # sample from all the labels based on the cropped patch (to balance pos and neg labels)
                neg_label_ratio_threshold = self.dataset_config[dataset_name]['neg_label_ratio_threshold']
                all_label_index_ls = [i for i in range(len(is_pos_in_crop))]
                if is_roi_crop :
                    # sample pos labels as many as possible (could be false pos?) will regrade if there is no positive
                    # chosen_label_index_ls: index of the chosen labels;
                    # is_pos_ls : True/False;
                    chosen_label_index_ls, is_pos_ls = self._select_pos_labels(all_label_index_ls, is_pos_in_crop, neg_label_ratio_threshold)   # [label1, label2, ....], [True, False, ...]
                else:
                    chosen_label_index_ls = random.sample(all_label_index_ls, min(self.max_queries, len(all_label_index_ls)))
                
                # so these are the chosen labels and their mask
                chosen_label = [labels[i] for i in chosen_label_index_ls]
                mask = mask[chosen_label_index_ls]
                if len(chosen_label) == self.max_queries:
                    chosen_label = [random.choice(ls) for ls in chosen_label]
                else:
                    chosen_label, source_indices = self.select_text_prompts(chosen_label)
                    indices = torch.tensor(source_indices)
                    mask = torch.index_select(mask, dim=0, index=indices)
                
                # NOTE: support nnUNet augmentation in CVPR25 challenge but still a BETA version
                if dataset_name in self.augmentator:
                    if not self.nnUNet_aug:
                        data_dict = {'image': image, 'label': mask}
                        aug_data_dict = self.augmentator[dataset_name](data_dict)
                        image, mask = aug_data_dict['image'], aug_data_dict['label']
                    else:
                        image = repeat(image.numpy(), 'c h w d -> b c d h w', b=1)
                        mask = repeat(mask.numpy(), 'c h w d -> b c d h w', b=1)
                        data_dict = {'data': image, 'seg': mask}
                        data_dict = self.augmentator[dataset_name](**data_dict)
                        image, mask = data_dict['data'], data_dict['target']
                        image = rearrange(image[0, ...], 'c d h w -> c h w d')
                        mask = rearrange(mask[0, ...], 'c d h w -> c h w d')  
                    
                if not isinstance(mask, torch.FloatTensor):
                    mask = mask.float()  # 转换为浮点类型
                    
                # simple check
                _, H, W, D = image.shape
                N, mH, mW, mD = mask.shape
                assert H == mH and W == mW and D == mD, f'image shape {H, W, D} inconsistent with mask shape {mH, mW, mD}'
                assert N == len(chosen_label), f'query num {len(chosen_label)} inconsistent with gt mask channels {N}'
                    
                break
            except SystemExit:
                exit()
            except:
                # record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {dataset_name} *** {data_path} ***\n')
                print(traceback_info)

        return {'image':image, 'mask':mask, 'text':chosen_label, 'modality':modality, 'image_path':data_path, 'mask_path':data_path, 'dataset':dataset_name, 'y1x1z1_y2x2z2':y1x1z1_y2x2z2}
    
if __name__ == '__main__':
    dataset = Med_SAM_Dataset(
        '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/train_10percent_raw_subset.jsonl', 
        '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/CVPR25_TextSegFMData_with_class.json',
        '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/data/dataset_config/cvpr25.json',
        crop_size=[288,288,96], 
        max_queries=16, 
        allow_repeat=True,
        nnUNet_aug=True
    )
    
    debug_dir = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/data/cvpr25_train_debug_visualization"
    os.makedirs(debug_dir, exist_ok=True)

    # For debugging, iterate over a fixed number of samples (e.g., 10)
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image, mask = sample["image"], sample["mask"]
        data_path = sample["image_path"]
        ds_name = sample['dataset']
        text = sample['text']
        basename = os.path.splitext(os.path.basename(data_path))[0]
        sample_dir = os.path.join(debug_dir, ds_name)
        os.makedirs(sample_dir, exist_ok=True)

        affine = np.eye(4)

        # Convert tensor to numpy array; remove singleton channel dimension if necessary
        img_np = image.numpy()
        if img_np.shape[0] == 1:
            img_np = img_np[0]
        nib.save(nib.Nifti1Image(img_np, affine), os.path.join(sample_dir, f"(img){basename}.nii.gz"))

        mask_np = mask.numpy()
        results = np.zeros((mask_np.shape[1], mask_np.shape[2], mask_np.shape[3])) # hwd
        for j in range(mask_np.shape[0]):
            results += mask_np[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)            # 每个label单独一个nii.gz
        nib.save(nib.Nifti1Image(results, affine), os.path.join(sample_dir, f"(seg){basename}.nii.gz"))
        
        with open(os.path.join(sample_dir, f"{basename}.txt"), 'w') as f:
            for i, t in enumerate(text):
                f.write(f'{i} : {t}\n')
        
        

    
    