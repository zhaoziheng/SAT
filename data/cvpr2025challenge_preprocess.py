import numpy as np
import json
from pathlib import Path
import os
import torch
import nibabel as nib
from tqdm import tqdm
import cv2
import shutil
import traceback
from einops import repeat, rearrange
import re
from datetime import datetime, timedelta

from data_loader_cvpr2025challenge import NAME2LOADER
import multiprocessing
import glob
              
def sc_mask_to_mc_mask(dataset_name, sample_name, sc_mask, label_values_ls):
    assert sc_mask.ndim == 3
    h, w, d = sc_mask.shape
    n = len(label_values_ls)
    mc_mask = np.zeros((n, h, w, d), dtype=np.uint8)
    for i, label_value in enumerate(label_values_ls):
        mc_mask[i] = (sc_mask == label_value)
    unique_intensities = np.unique(sc_mask).tolist()
    for intensity in unique_intensities:
        if intensity != 0 and int(intensity) not in label_values_ls:
            Path(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}').mkdir(exist_ok=True, parents=True)
            beijing_time = datetime.utcnow() + timedelta(hours=8)
            formatted_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S")
            with open(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/warning.txt', 'a') as f:
                f.write(f'Beijing Time: {formatted_time}\n')
                f.write(f'** {dataset_name} **')
                f.write(f"Intensity {intensity} found in sc_mask is not in label_values_ls\n\n")
    return mc_mask
    
def check_sample(npz_file_path):
    try:
        data = np.load(npz_file_path)
    except Exception as e:
        print(f'Cant Load {npz_file_path}')
        print(e)
        return
    
    # identify dataset
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/dataset_prefixes.json', 'r') as f:
        dataset_prefixes = json.load(f)
        
    sample_name = os.path.basename(npz_file_path)[:-4]
    found = False
    for prefix, dataset in dataset_prefixes.items():
        if sample_name.startswith(prefix):
            dataset_name = dataset
            found = True
    if not found:
        print(f'{sample_name} not in CVPR25_TextSegFMData_with_class.json')
        dataset_name = 'Unknown'
    
    # dataset_name = Path(npz_file_path).parent.name
    modality = Path(npz_file_path).parent.parent.name
    sample_name = os.path.basename(npz_file_path)[:-4]
    
    img_tensor = data['imgs'].astype(np.float32)  # 0~255
    if 'gts' in data:
        sc_mask = data['gts'].astype(np.float32)
    else:
        sc_mask = np.zeros_like(img_tensor)
    spacing = data['spacing']

    # adjust plane
    spacing = spacing.tolist()
    # print(f'** {dataset_name} ** Original spacing: {spacing} Image Shape {img_tensor.shape}')
    # if not (spacing[0] == spacing[1] == spacing[2]):
    #     if spacing[0] == max(spacing):
    #         # d h w -> h w d
    #         img_tensor = rearrange(img_tensor, 'd h w -> h w d')
    #         sc_mask = rearrange(sc_mask, 'd h w -> h w d')
    #         spacing.append(spacing.pop(0))
    #     elif spacing[1] == max(spacing):
    #         # d h w -> h w d
    #         img_tensor = rearrange(img_tensor, 'h d w -> h w d')
    #         sc_mask = rearrange(sc_mask, 'h d w -> h w d')
    #         spacing.append(spacing.pop(1))
    # print(f'** {dataset_name} ** Transferred spacing: {spacing} Image Shape {img_tensor.shape}')
    img_tensor, sc_mask, spacing = NAME2LOADER[dataset_name](sample_name, img_tensor, sc_mask, spacing)
    
    # 检查 img_tensor 中最短的维度是否位于最后一个位置
    tensor_shape = img_tensor.shape
    min_dim_index = tensor_shape.index(min(tensor_shape))
    if min_dim_index != len(tensor_shape) - 1 and not (tensor_shape[0]==tensor_shape[1]==tensor_shape[2]):
        Path(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}').mkdir(exist_ok=True, parents=True)
        beijing_time = datetime.utcnow() + timedelta(hours=8)
        formatted_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/warning.txt', 'a') as f:
            f.write(f'Beijing Time: {formatted_time}\n')
            f.write(f'** {dataset_name} **')
            f.write(f"警告: img_tensor 最短的维度不在最后一个位置。当前形状: {tensor_shape} Spacing: {spacing}。\n\n")

    # 检查 spacing 中最大的元素是否位于最后一个位置
    max_spacing_index = np.argmax(spacing)
    if max_spacing_index != len(spacing) - 1 and not (spacing[0]==spacing[1]==spacing[2]):
        Path(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}').mkdir(exist_ok=True, parents=True)
        beijing_time = datetime.utcnow() + timedelta(hours=8)
        formatted_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/warning.txt', 'a') as f:
            f.write(f'Beijing Time: {formatted_time}\n')
            f.write(f'** {dataset_name} **')
            f.write(f"警告: spacing 中最大的元素不在最后一个位置。当前形状: {tensor_shape} Spacing: {spacing}。\n\n")
    
    if dataset_name != 'Unknown':
        with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/CVPR25_TextSegFMData_with_class.json', 'r') as f:
            label_2_text_prompt = json.load(f)
        label_2_text_prompt = label_2_text_prompt[dataset_name] # "1": ["White matter hyperintensities", ...], "2": [...]
        
        label_values_ls = list(label_2_text_prompt.keys())
        label_values_ls = [int(v) for v in label_values_ls if v!='instance_label']
        
        label_name_ls = list(label_2_text_prompt.values())
        text_ls = [ls[0] for ls in label_name_ls if isinstance(ls, list)]
        
        mc_mask = sc_mask_to_mc_mask(dataset_name, sample_name, sc_mask, label_values_ls)
        
    else:
        text_ls = []        

    # 检查数据
    print('* file path * : ', npz_file_path)
    print('* modality * : ', modality)
    print('* labels * : ', text_ls)
    print('* spacing * : ', spacing)
    print('* img_tensor.shape * : ', img_tensor.shape)  # [c h w d]
    print('* img_tensor.range * : ', np.min(img_tensor), np.max(img_tensor))  # [c h w d]
    print('* img_tensor.dtype * : ', img_tensor.dtype)
    if dataset_name != 'Unknown':
        print('* mc_mask.shape * : ', mc_mask.shape)    # [c h w d]
        print('* mc_mask.dtype * : ', mc_mask.dtype)
        print('* sum(mc_mask) * : ', np.sum(mc_mask))
    
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max()-img_tensor.min())
    Path(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/segmentations').mkdir(exist_ok=True, parents=True)
    
    # 3D按nifiti存
    results = np.zeros((img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])) # hwd
    for j, label in enumerate(text_ls):
        results += mc_mask[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
        # 每个label单独一个nii.gz
        segobj = nib.nifti2.Nifti1Image(mc_mask[j, :, :, :], np.eye(4))
        nib.save(segobj, f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/segmentations/{label}.nii.gz')
    segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
    nib.save(segobj, f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/seg.nii.gz')
    
    imgobj = nib.nifti2.Nifti1Image(img_tensor, np.eye(4))   # hwd
    nib.save(imgobj, f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/img.nii.gz')

    # 按slice存
    for slice_idx in tqdm(range(img_tensor.shape[-1])):
        Path(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/%s/%s/slice_%d'%(dataset_name, sample_name, slice_idx)).mkdir(parents=True, exist_ok=True)
        Path(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/%s/%s/image_series'%(dataset_name, sample_name)).mkdir(parents=True, exist_ok=True)
        img = repeat(img_tensor[:, :, slice_idx], 'h w -> h w c', c=1) # [H, W, C]
        cv2.imwrite(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/%s/%s/slice_%d/img.jpg'%(dataset_name, sample_name, slice_idx), img*255.0)
        cv2.imwrite(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/%s/%s/image_series/slice_%d.jpg'%(dataset_name, sample_name, slice_idx), img*255.0)
        for label_idx, text in enumerate(text_ls):
            msk = mc_mask[label_idx, :, :, slice_idx]
            if np.sum(msk) > 0:
                """
                # the bbox
                non_zero_coordinates = np.nonzero(msk) # ([...], [...])
                y1, x1 = np.min(non_zero_coordinates[0]).item(), np.min(non_zero_coordinates[1]).item()
                y2, x2 = np.max(non_zero_coordinates[0]).item(), np.max(non_zero_coordinates[1]).item()
                print('slice no.%d, label no.%d : %s, [x1, y1, x2, y2] : [%d, %d, %d, %d]'%(slice_idx, label_idx, text, x1, y1, x2, y2))
                """
                cv2.imwrite(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/%s/%s/slice_%d/%d_%s_msk.jpg'%(dataset_name, sample_name,  slice_idx, label_idx, text), msk*255.0)
                if img.shape[2] == 1:
                    img = repeat(img, 'h w c -> h w (c r)', r=3)
                overlap = repeat(msk, 'h w -> h w c', c=3) # colorful mask H, W, C
                img = np.float32(img)
                overlap = np.float32(overlap)
                overlap = cv2.add(img, overlap*255.0)
        
    spacing = ", ".join(map(str, spacing))        
    with open(f'/mnt/hwfile/medai/zhaoziheng/SAM/visualization/CVPR25_Challenge/{dataset_name}/{sample_name}/spacing.txt', 'w') as f:
        f.write(spacing)
             
def load_npz_data(label_json, dataset_name, data_path):
    data = np.load(data_path)
    sample_name = os.path.basename(data_path)[:-4]
    
    img = data['imgs'].astype(np.float32)  # 0~255
    sc_mask = data['gts'].astype(np.float32)
    spacing = data['spacing'].tolist()
    
    img, sc_mask, spacing = NAME2LOADER[dataset_name](sample_name, img, sc_mask, spacing)
    img = img[np.newaxis, :, :, :]  # 1 h w d
    
    label_2_text_prompt = label_json[dataset_name] # '1':['xxx', 'xxx', ...]
    label_values_ls = list(label_2_text_prompt.keys())
    label_values_ls = [int(v) for v in label_values_ls if v!='instance_label']
    text_prompt_ls = list(label_2_text_prompt.values()) # list of list of str
    text_prompt_ls = [ls for ls in text_prompt_ls if isinstance(ls, list)]
    
    mc_mask = sc_mask_to_mc_mask(sc_mask, label_values_ls)  # n h w d
    
    return img, mc_mask, text_prompt_ls

def check_label_parallel(raw_jsonl, label_json, out_jsonl):
    """
    check label for each dataset and then merge
    """
    
    dataset_names = set()
    with open(raw_jsonl, 'r') as f:
        for line in f:
            datum = json.loads(line.strip())
            if 'dataset' in datum:
                dataset_names.add(datum['dataset'])
    print("Unique dataset values:", list(dataset_names))
    
    def process_datasets(sub_datasets):
        for ds in sub_datasets:
            updated = check_label(raw_jsonl, label_json, ds)
            out_path = f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/tmp/{ds}.jsonl'
            with open(out_path, 'w') as f:
                for datum in updated:
                    f.write(json.dumps(datum) + '\n')

    all_datasets = list(dataset_names)
    n_processes = 8
    chunk_size = (len(all_datasets) + n_processes - 1) // n_processes
    chunks = [all_datasets[i*chunk_size:(i+1)*chunk_size] for i in range(n_processes)]

    processes = []
    for sub in chunks:
        p = multiprocessing.Process(target=process_datasets, args=(sub,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(out_jsonl, 'w') as fout:
        for fp in glob.glob('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/tmp/*.jsonl'):
            if fp == out_jsonl:
                continue
            with open(fp, 'r') as fin:
                for line in fin:
                    fout.write(line)
                    
    for tmp_file in glob.glob('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/tmp/*.jsonl'):
        os.remove(tmp_file)
        
def check_label(jsonl, label_json, target_dataset):
    """
    for each sample (in trainset)
    check each class whether annotated on the gt mask
    """
    
    with open(jsonl, 'r') as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    
    with open(label_json, 'r') as f:
        label_json = json.load(f)
    
    processed_data = []
    for sample in tqdm(lines):
        dataset_name = sample['dataset']
        if dataset_name != target_dataset:
            continue
        
        data_path = sample['data']
        try:
            image, mask, labels = load_npz_data(label_json, dataset_name, data_path)
        except Exception as e:
            beijing_time = datetime.utcnow() + timedelta(hours=8)
            formatted_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S")
            with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/load_error.txt', 'a') as log_file:
                log_file.write(f"[{formatted_time}] data_path: {data_path}, error: \n{str(e)}\n\n")
            continue
        
        assert mask.shape[0] == len(labels), f'{mask.shape} != {labels}'
        
        label_existance = []
        for i, label in enumerate(labels):
            if np.any(mask[i]):
                label_existance.append(True)
            else:
                label_existance.append(False)
        
        processed_sample = {k:v for k,v in sample.items()}
        processed_sample['label_existance'] = label_existance
        processed_data.append(processed_sample)
            
    return processed_data

def merge_validation_data():
    """
    merge validation imgs and gts into one npz file
    """
    for sample_name in os.listdir('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_npz'):
        if sample_name.startswith('._') or not sample_name.endswith('.npz') or os.path.exists(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_npz_new/{sample_name}'):
            print(sample_name)
            continue
        assert os.path.exists(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_gt/{sample_name}'), sample_name
        data = dict(np.load(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_npz/{sample_name}', allow_pickle=True))
        gt = np.load(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_gt/{sample_name}')
        assert gt['spacing'].tolist() == data['spacing'].tolist(), sample_name
        assert gt['gts'].shape == data['imgs'].shape, sample_name
        data['gts'] = gt['gts']
        np.savez(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_npz_new/{sample_name}', **data)

def prepare_validation_data():
    """
    collect validation data into jsonl file
    make sure call merge_validation_data first
    """

    data = []
    
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/CVPR25_TextSegFMData_with_class.json', 'r') as f:
        label_2_text_prompt = json.load(f)
        
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/dataset_prefixes.json', 'r') as f:
        dataset_prefixes = json.load(f)
        
    all_datasets = list(label_2_text_prompt.keys())
        
    for filename in os.listdir('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_npz_new'):
        if filename.endswith('.npz') and not filename.startswith('._'):
            # identify modality
            modality = filename.split('_')[0]
            # identify dataset
            found = False
            for prefix, dataset_name in dataset_prefixes.items():
                if filename.startswith(prefix):
                    dataset = dataset_name
                    found = True
            if not found:
                print(f'{filename} not in CVPR25_TextSegFMData_with_class.json')
                continue
                
            # collect
            data.append({'data':f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_val_npz_new/{filename}', 'dataset':dataset, 'modality':modality})
                
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/validation(filtered).jsonl', 'w') as f:
        for datum in data:
            f.write(json.dumps(datum)+'\n')
            
def prepare_train_data():
    """
    collect train data into jsonl file
    """
    
    data = []
    
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/CVPR25_TextSegFMData_with_class.json', 'r') as f:
        label_2_text_prompt = json.load(f)
        
    for modality in os.listdir('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_train_npz_random_10percent_16G'):
        if modality not in ['CT', 'MR', 'PET', 'US3D', 'Microscopy']:
            continue
        
        for dataset_name in os.listdir(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_train_npz_random_10percent_16G/{modality}'):
            if dataset_name not in label_2_text_prompt:
                print(dataset_name)
                continue
            else:
                for filename in os.listdir(f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_train_npz_random_10percent_16G/{modality}/{dataset_name}'): 
                    if filename.endswith('.npz') and not filename.startswith('.'):
                        data.append({'data':f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/3D_train_npz_random_10percent_16G/{modality}/{dataset_name}/{filename}', 'dataset':dataset_name, 'modality':modality})
                        
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/train.jsonl', 'w') as f:
        for datum in data:
            f.write(json.dumps(datum)+'\n')
                    
def find_dataset_start(jsonl='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/train_10percent.jsonl'):
    with open(jsonl, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    
    dataset_filenames = {}
    for datum in data:
        dataset_name = datum['dataset']
        if dataset_name not in dataset_filenames:
            dataset_filenames[dataset_name] = []
        dataset_filenames[dataset_name].append(datum['data'])
    
    common_prefix_to_dataset_name = {}
    for dataset_name, filenames in dataset_filenames.items():
        # Extract filenames without paths
        base_filenames = [os.path.basename(f) for f in filenames]
        
        # Find common prefixes if any
        if len(base_filenames) > 1:
            # Sort to make similar filenames adjacent
            base_filenames.sort()
            
            # Find common prefix
            first = base_filenames[0]
            last = base_filenames[-1]
            
            # Find how many characters match at the start
            common_prefix = ""
            for i in range(min(len(first), len(last))):
                if first[i] == last[i]:
                    common_prefix += first[i]
                else:
                    break
            
            if common_prefix in common_prefix_to_dataset_name:
                print(f"Common prefix {common_prefix} found in multiple datasets: {dataset_name} and {common_prefix_to_dataset_name[common_prefix]}")
                continue
            
            common_prefix_to_dataset_name[common_prefix] = dataset_name
            
    with open('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/dataset_prefixes.json', 'w') as f:
        json.dump(common_prefix_to_dataset_name, f, indent=4)
        
            
def simplify_json(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each dataset
    simplified_data = {}
    for dataset, classes in data.items():
        simplified_data[dataset] = {}
        for class_id, descriptions in classes.items():
            # Keep only the first element if it's a list
            if isinstance(descriptions, list) and descriptions:
                simplified_data[dataset][class_id] = [descriptions[0]]
            else:
                simplified_data[dataset][class_id] = descriptions
    
    # Save the simplified JSON
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=4)
    
    print(f"Created simplified JSON at {output_file}")
    
def summarize_dataset_distribution(jsonl_path, key, statistic_json):
    """
    statistic train-validation split
    """
    if os.path.exists(statistic_json):
        with open(statistic_json, 'r') as f:
            dataset_distribution = json.load(f)
    else:
        dataset_distribution = {}
    
    # Read the jsonl file and count datasets
    with open(jsonl_path, 'r') as f:
        for line in f:
            datum = json.loads(line)
            if datum['dataset'] not in dataset_distribution:
                dataset_distribution[datum['dataset']] = {key:0}
            if key not in dataset_distribution[datum['dataset']]:
                dataset_distribution[datum['dataset']][key] = 0
            dataset_distribution[datum['dataset']][key] += 1

    # Save the distribution to a JSON file
    with open(statistic_json, 'w') as f:
        json.dump(dataset_distribution, f, indent=4)
    
    return dataset_distribution

# # Example usage
# jsonl_path = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/train_10percent.jsonl"
# output_path = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/data_distribution.json"
# distribution = summarize_dataset_distribution(jsonl_path, 'train_10percent', output_path)
# print(f"Distribution saved to {output_path}")

# jsonl_path = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/train.jsonl"
# output_path = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/data_distribution.json"
# distribution = summarize_dataset_distribution(jsonl_path, 'train', output_path)
# print(f"Distribution saved to {output_path}")

# jsonl_path = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/validation.jsonl"
# output_path = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/data_distribution.json"
# distribution = summarize_dataset_distribution(jsonl_path, 'validation', output_path)
# print(f"Distribution saved to {output_path}")

if __name__ == '__main__':
    
    # # Check Label
    
    # check_label_parallel(
    #     raw_jsonl='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/challenge_data/train_10percent.jsonl', 
    #     label_json='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/CVPR25_TextSegFMData_with_class.json', 
    #     out_jsonl='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/data/challenge_data/train_10percent.jsonl'
    # )
    
    check_sample('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/inputs/CT_demo1.npz')
    check_sample('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/inputs/CT_demo2.npz')
    check_sample('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/inputs/MR_demo.npz')
    check_sample('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT/inputs/US_demo.npz')