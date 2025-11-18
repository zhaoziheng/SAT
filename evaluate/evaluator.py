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
import pickle
from scipy.ndimage import gaussian_filter
import torch.distributed as dist

from evaluate.metric import calculate_metric_percase
from evaluate.merge_after_evaluate import merge
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

def evaluate(model, 
             text_encoder, 
             device, 
             testset, 
             testloader, 
             dice_score,
             nsd_score,
             csv_path, 
             resume,
             save_interval,
             visualization):
    
    # if to store pred、gt、img (as nii.gz
    if visualization:
        nib_dir = csv_path.replace('.csv', '')
        
    # collate in master process
    if is_master():
        shutil.copy(testset.jsonl_file, csv_path.replace('.csv', '.jsonl'))
        
        # datasets --> labels --> metrics
        datasets_labels_metrics = {}   # {'COVID19':{'covid19_infection':{'dice':[0.8, 0.9, ...], ...} ...}, ...}
    
        # datasets --> samples --> labels --> metrics
        samples_labels_metrics = {}   # {'COVID19':{'0.npy':{'covid19_infection':{'dice':0.8, ...} ...}, ...} 记录每个dataset里的sample（行）
        
        # datsets --> labels
        datasets_labels_sets = {}    # {'COVID19':set('covid19_infection', ...), ...}  记录每个dataset里的label种类（列）
        
    # accumulate scores of each sample in each process
    results_of_samples = [] # each element : [dataset_name, modality, sample_id, scores_of_labels(dict), label_names] 
    
    # load results from an interrupted eval (only in master process)
    if resume and is_master():
        root_dir = os.path.dirname(csv_path)
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')  # xxx/test/step_xxx.csv --> step_xxx_tmp_rank
        pkl_to_del = []
        for f in os.listdir(root_dir):
            if prefix in f:
                # load list of results
                pkl_path = f'{root_dir}/{f}'
                with open(pkl_path, 'rb') as f:
                    results_of_samples += pickle.load(f)
                print(f'Load results from {pkl_path}')   
                pkl_to_del.append(pkl_path)
                
        # there may be duplication? We leave the deduplication to the final merge
        # merge all the loaded samples, del the tmp pickle files in previous evaluation task
        for pkl_path in pkl_to_del:
            os.remove(pkl_path)
            print(f'Del {pkl_path}')
        merge_pkl = csv_path.replace('.csv', f'_tmp_rank0.pkl')
        with open(merge_pkl, 'wb') as f:
            pickle.dump(results_of_samples, f)  
        print(f'Load results of {len(results_of_samples)} samples, Merge into {merge_pkl}')
                        
    model.eval()
    text_encoder.eval()
        
    with torch.no_grad():
        
        data_time = 0
        pred_time = 0
        metric_time = 0
        
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
        for sample in testloader:    # in evaluation/inference, a "batch" in loader is a volume
            # data loading
            dataset_name = sample['dataset_name']
            sample_id = sample['sample_id'] 
            batched_patches = sample['batched_patches']
            batched_y1y2_x1x2_z1z2 = sample['batched_y1y2_x1x2_z1z2']
            split_labels = sample['split_labels'] 
            split_n1n2 = sample['split_n1n2']
            gt_segmentation = sample['gt_segmentation'].numpy()  # n h w d
            labels = sample['labels']
            modality = sample['modality']
            image_path = sample['image_path']

            n,h,w,d = gt_segmentation.shape
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
            
            pred_time += (time.time()-end_time)
            end_time = time.time()
                            
            # avg            
            prediction = prediction / accumulation
            prediction = torch.where(prediction>0.5, 1.0, 0.0)
            prediction = prediction.numpy()
            
            # cal metrics : [{'dice':x, ...}, ...] 
            scores = []
            for j in range(len(labels)):
                scores.append(calculate_metric_percase(prediction[j, :, :, :], gt_segmentation[j, :, :, :], dice_score, nsd_score))    # {'dice':0.9, 'nsd':0.8} 每个label一个dict
            
            # visualization  
            if visualization:
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
                
                image = testset.load_image(image_path)
                image = np.squeeze(image)
                imgobj = nib.nifti2.Nifti1Image(image, np.eye(4))
                nib.save(imgobj, f'{nib_dir}/{dataset_name}/img_{sample_id}.nii.gz')
                
                gt = np.zeros((h, w, d)) # hwd
                for j, label in enumerate(labels):
                    gt += gt_segmentation[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
                    Path(f'{nib_dir}/{dataset_name}/gt_{sample_id}').mkdir(exist_ok=True, parents=True)
                    # 每个label单独一个nii.gz
                    segobj = nib.nifti2.Nifti1Image(gt_segmentation[j, :, :, :], np.eye(4))
                    nib.save(segobj, f'{nib_dir}/{dataset_name}/gt_{sample_id}/{label}.nii.gz')
                gtobj = nib.nifti2.Nifti1Image(gt, np.eye(4))
                nib.save(gtobj, f'{nib_dir}/{dataset_name}/gt_{sample_id}.nii.gz')
                
            metric_time += (time.time()-end_time)
            end_time = time.time()
            
            # accumulate
            results_of_samples.append([dataset_name, modality, sample_id, scores, labels])
            
            # save in each process regularly in case of interruption
            if len(results_of_samples) % save_interval == 0:
                with open(csv_path.replace('.csv', f'_tmp_rank{dist.get_rank()}.pkl'), 'wb') as f:
                    pickle.dump(results_of_samples, f)
            
        """
        # gather results from all device to rank-0 (solution 1)
        gather_results = [None for i in range(dist.get_world_size())]
        dist.gather_object(
            results_of_samples, 
            gather_results if dist.get_rank() == 0 else None,
            dst = 0
            )
        
        if int(dist.get_rank()) == 0:
            results_of_samples = [tmp for ls in results_of_samples for tmp in ls]
        """         
                
        avg_patch_batch_num /= len(testloader)
        avg_query_batch_num /= len(testloader)
        data_time /= len(testloader)
        pred_time /= len(testloader)
        metric_time /= len(testloader)
        print(f'On Rank {dist.get_rank()}, each sample has {avg_patch_batch_num} batch of patches and {avg_query_batch_num} batch of queries, Data Time: {data_time}, Pred Time: {pred_time}, Dice Time: {metric_time}')
          
        torch.cuda.empty_cache()
        
        # save in each process (to a fnl pickle, also denoting this process ends)
        with open(csv_path.replace('.csv', f'_fnl_rank{dist.get_rank()}.pkl'), 'wb') as f:
            pickle.dump(results_of_samples, f)
                    
        # gather and record in rank 0 (solution 2)
        if is_master():
            
            # detect the finish of each process
            while True:
                all_process_finished = True
                for rank_id in range(torch.distributed.get_world_size()):
                    if not os.path.exists(csv_path.replace('.csv', f'_fnl_rank{rank_id}.pkl')): # xxx_tmp_rankx.pkl
                        all_process_finished = False
                        break
                if all_process_finished:
                    break
                else:
                    time.sleep(10)
            
            # read results of each process (samples may be duplicated due to the even distribution of ddp, check)
            results_of_samples = []    
            for rank_id in range(torch.distributed.get_world_size()):
                fnl_results_file = csv_path.replace('.csv', f'_fnl_rank{rank_id}.pkl')
                tmp_results_file = csv_path.replace('.csv', f'_tmp_rank{rank_id}.pkl')
                with open(fnl_results_file, 'rb') as f:
                    results_of_samples += pickle.load(f)
                os.remove(fnl_results_file)
                if os.path.exists(tmp_results_file):
                    os.remove(tmp_results_file)
                
            # check duplication
            unique_set = set()
            deduplicated_results_of_samples = []
            for dataset_name, modality, sample_id, scores, labels in results_of_samples:
                if f'{dataset_name}/{sample_id}' not in unique_set:
                    unique_set.add(f'{dataset_name}/{sample_id}')
                    deduplicated_results_of_samples.append([dataset_name, modality, sample_id, scores, labels])
            results_of_samples = deduplicated_results_of_samples
            
            # save for tmp
            with open(csv_path.replace('.csv', '.pkl'), 'wb') as f:
                pickle.dump(results_of_samples, f)

            # collate results
            for dataset_name, modality, sample_id, scores, labels in results_of_samples:    #  [[dataset_name, modality, sample_id, scores_of_labels(dict), label_names], ...]
                dataset_name = f'{dataset_name}({modality})'
                
                if dataset_name not in datasets_labels_metrics:
                    datasets_labels_metrics[dataset_name] = {}  # {'COVID19(CT)':{}}
                if dataset_name not in datasets_labels_sets:
                    datasets_labels_sets[dataset_name] = set()  # {'COVID19(CT)':set()}
                if dataset_name not in samples_labels_metrics:
                    samples_labels_metrics[dataset_name] = {}
                samples_labels_metrics[dataset_name][sample_id] = {}   # {'COVID19(CT)':{'0':{}}}
                
                for metric_dict, label in zip(scores, labels):
                    # accumulate metrics （for per dataset per class
                    # {'COVID19(CT)':{'covid19_infection':{'dice':[0.8, 0.9, ...], 'nsd':[0.8, 0.9, ...], ...} ...}, ...}
                    if label not in datasets_labels_metrics[dataset_name]:
                        datasets_labels_metrics[dataset_name][label] = {k:[v] for k,v in metric_dict.items()}
                    else:
                        for k,v in metric_dict.items():
                            datasets_labels_metrics[dataset_name][label][k].append(v)
                    
                    # statistic labels
                    # {'COVID19(CT)':set('covid19_infection', ...)}
                    if label not in datasets_labels_sets[dataset_name]:
                        datasets_labels_sets[dataset_name].add(label)
                    
                    # record metrics （for per dataset per sample per class
                    # {'COVID19':{'0.npy':{'covid19_infection':{'dice':0.8, 'nsd':0.9, ...} ...}, ...}
                    samples_labels_metrics[dataset_name][sample_id][label] = {k:v for k,v in metric_dict.items()}
                        
            # average and log (列为metrics，例如dice，nsd...)
            # create a df like:
            # {
            #   'TotalSegmentator': [0.xx, 0.xx, ...]    # 在T之前，这是一列
            #   'TotalSegmentator, Lung': [0.68, 0.72, ...]
            # }
            # by defult, print the dice (1st metric) of each dataset 
            info = 'Metrics of Each Dataset:\n'
            avg_df = {}
            for dataset in datasets_labels_metrics.keys():
                avg_df[dataset] = {k:[] for k in metric_dict.keys()}    # 'TotalSegmentator(CT)': {'dice':[0.8, ...] 'nsd':[0.5, ...], ...}
                for label in datasets_labels_metrics[dataset].keys():
                    avg_df[f'{dataset}, {label}'] = []
                    for metric in datasets_labels_metrics[dataset][label].keys():
                        label_metric = np.average(datasets_labels_metrics[dataset][label][metric])
                        avg_df[f'{dataset}, {label}'].append(label_metric)  # 'TotalSegmentator, Lung': [0.68, 0.72, ...] list of num_metrics
                        avg_df[dataset][metric].append(label_metric)
                avg_df[dataset] = {k:np.average(v) for k,v in avg_df[dataset].items()} # 'TotalSegmentator': {'dice':[0.8, ...] 'nsd':[0.5, ...], ...} --> 'TotalSegmentator': {'dice':0.x, 'nsd':0.x, ...}
                info += f'{dataset}  |  ' 
                for k ,v in avg_df[dataset].items():
                    info += f'{v}({k})  |  '
                info += '\n'
                avg_df[dataset] = list(avg_df[dataset].values())
            avg_df = pd.DataFrame(avg_df).T
            avg_df.columns = list(metric_dict.keys())   # ['dice', 'nsd']
            avg_df.to_csv(csv_path)        
            print(info)
            
            # detailed log （nsd和dice，列为class label
            # multi-sheet, two for each dataset
            df_list = [['summary', avg_df]]
            for dataset, label_set in datasets_labels_sets.items():
                metric_df ={}
                if dice_score:
                    metric_df['dice'] = {}
                if nsd_score:
                    metric_df['nsd'] = {}

                # create dfs like:
                # {
                #   '0.npy': [0.xx, 0.xx, ...]
                #   ......
                # }
                
                # {'COVID19':{'0.npy':{'covid19_infection':{'dice':0.8, ...} ...}, ...}
                for image_id, label_dict in samples_labels_metrics[dataset].items():
                    for metric in metric_df:
                        tmp = []    # one dice for each label in this dataset
                        for label in label_set:
                            score = label_dict[label][metric] if label in label_dict else -1
                            tmp.append(score)
                        metric_df[metric][image_id] = tmp   
                
                for metric, metric_df in metric_df.items():
                    metric_df = pd.DataFrame(metric_df).T
                    metric_df.columns = list(label_set)
                    df_list.append([dataset+f'({metric})', metric_df])
                
            xlsx_path = csv_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(xlsx_path) as writer:
                for name, df in df_list:
                    # 将每个 DataFrame 写入一个 sheet(sheet name must be < 31)
                    if len(name) > 31:
                        name = name[len(name)-31:]
                    df.to_excel(writer, sheet_name=name, index=True)    
                    
            # avg_dice_over_merged_labels, avg_nsd_over_merged_labels = merge(region_split_json, label_statistic_json, xlsx_path, xlsx_path)  
            
            os.remove(csv_path.replace('.csv', '.pkl'))
            
        else:
            
            pass
        
            # avg_dice_over_merged_labels = avg_nsd_over_merged_labels = 0
            
        return # avg_dice_over_merged_labels, avg_nsd_over_merged_labels
                    
            