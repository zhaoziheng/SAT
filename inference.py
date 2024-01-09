import os
import datetime
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
# from scipy.ndimage import gaussian_filter
from einops import reduce, rearrange, repeat

from dataset.inference_dataset import Inference_Dataset, inference_collate_fn
from model.tokenizer import MyTokenizer
from model.build_model import load_text_encoder, build_segmentation_model

def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    # build an gaussian filter with the patch size
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def inference(model, tokenizer, text_encoder, device, testloader):
    # inference
    model.eval()
    text_encoder.eval()
        
    with torch.no_grad():
        
        # in ddp, only master process display the progress bar
        if int(os.environ["LOCAL_RANK"]) != 0:
            testloader = tqdm(testloader, disable=True)
        else:
            testloader = tqdm(testloader, disable=False)  
        
        # gaussian kernel to accumulate predcition
        gaussian = np.ones((288, 288, 96)) # compute_gaussian((288, 288, 96))

        for batch in testloader:    # one batch for each sample
            # data loading
            batched_patch = batch['batched_patch']
            batched_y1y2x1x2z1z2 = batch['batched_y1y2x1x2z1z2']
            split_prompt = batch['split_prompt'] 
            split_n1n2 = batch['split_n1n2']
            labels = batch['labels']
            image = batch['image']
            image_path = batch['image_path']
            
            _,h,w,d = image.shape
            n = split_n1n2[-1][-1]
            prediction = np.zeros((n, h, w, d))
            accumulation = np.zeros((n, h, w, d))
            
            # for each batch of queries
            for prompts, n1n2 in zip(split_prompt, split_n1n2):
                n1, n2 = n1n2
                input_text = tokenizer.tokenize(prompts) # (max_queries, max_l)
                input_text['input_ids'] = input_text['input_ids'].to(device=device)
                input_text['attention_mask'] = input_text['attention_mask'].to(device=device)
                queries, _, _ = text_encoder(text1=input_text, text2=None) # (max_queries, d)
                
                # for each batch of patches
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patch, batched_y1y2x1x2z1z2):   # [b, c, h, w, d] 
                    batched_queries = repeat(queries, 'n d -> b n d', b=patches.shape[0])   # [b, n, d]
                    patches = patches.to(device=device)
                    
                    prediction_patch = model(queries=batched_queries, image_input=patches)
                    prediction_patch = torch.sigmoid(prediction_patch)
                    prediction_patch = prediction_patch.detach().cpu().numpy()  # bnhwd
                    
                    # fill in 
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]

                        # gaussian accumulation
                        prediction[n1:n2, y1:y2, x1:x2, z1:z2] += prediction_patch[b, :n2-n1, :y2-y1, :x2-x1, :z2-z1] * gaussian[:y2-y1, :x2-x1, :z2-z1]
                        accumulation[n1:n2, y1:y2, x1:x2, z1:z2] += gaussian[:y2-y1, :x2-x1, :z2-z1]
                            
            # avg            
            prediction = prediction / accumulation
            prediction = np.where(prediction>0.5, 1.0, 0.0)
            
            # save prediction
            save_dir = image_path.split('.')[0] # xxx/xxx.nii.gz --> xxx/xxx
            np_images = image.numpy()[0, :, :, :]
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            results = np.zeros((h, w, d)) # merge on one channel
            for j, label in enumerate(labels):
                results += prediction[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
                pred_obj = nib.nifti2.Nifti1Image(prediction[j, :, :, :], np.eye(4))
                nib.save(pred_obj, f'{save_dir}/{label}.nii.gz')
            pred_obj = nib.nifti2.Nifti1Image(results, np.eye(4))
            nib.save(pred_obj, f'{save_dir}/prediction.nii.gz')
            
            # save image
            imgobj = nib.nifti2.Nifti1Image(np_images, np.eye(4))
            nib.save(imgobj, f'{save_dir}/image.nii.gz')

def main(args):
    # set gpu
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200)) # might takes a long time to sync between process
    # dispaly
    if int(os.environ["LOCAL_RANK"]) == 0:
        print('** GPU NUM ** : ', torch.cuda.device_count())  # 打印gpu数量
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    print(f"** DDP ** : Start running on rank {rank}.")
    
    # dataset and loader
    testset = Inference_Dataset(args.data_jsonl, args.patch_size, args.max_queries, args.batchsize)
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=sampler, batch_size=1, collate_fn=inference_collate_fn, shuffle=False)
    sampler.set_epoch(0)
    
    # set segmentation model
    model = build_segmentation_model(args, device, gpu_id)
    
    # load checkpoint of segmentation model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if int(os.environ["RANK"]) == 0:
        print(f"** Model ** Load segmentation model from {args.checkpoint}.")
    
    # load text encoder
    text_encoder = load_text_encoder(args, device, gpu_id)
    
    # set tokenizer
    tokenizer = MyTokenizer(args.tokenizer_path)
    
    # choose how to evaluate the checkpoint
    inference(model, tokenizer, text_encoder, device, testloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path of SAT",
    )
    parser.add_argument(
        "--text_encoder_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path of text encoder",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_jsonl",
        type=str,
        help="Path to jsonl file",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs='+',
        default=[288, 288, 96],
        help='Size of input patch',
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default='UNET',
    )
    args = parser.parse_args()
    
    main(args)