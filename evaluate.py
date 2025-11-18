import os
import datetime
import random
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torch.distributed as dist

from data.evaluate_dataset import Evaluate_Dataset, Evaluate_Dataset_OnlineCrop, collate_fn
from model.build_model import build_maskformer, load_checkpoint
from model.text_encoder import Text_Encoder
from evaluate.evaluator import evaluate
from evaluate.params import parse_args
from train.dist import is_master

def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main(args):
    # set gpu
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=10800))   # might takes a long time to sync between process
    
    # dispaly
    if is_master():
        print('** GPU NUM ** : ', torch.cuda.device_count())  # 打印gpu数量
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    rank = dist.get_rank()
    print(f"** DDP ** : Start running DDP on rank {rank}.")
    
    # file to save the detailed metrics
    csv_path = f'{args.rcd_dir}/{args.rcd_file}.csv'
    txt_path = f'{args.rcd_dir}/{args.rcd_file}.txt'
    if is_master():
        Path(args.rcd_dir).mkdir(exist_ok=True, parents=True)
        print(f'Detailed Results will be Saved to {csv_path} and {txt_path}')
        
    # resume an evaluation if specified
    evaluated_samples = set()
    if args.resume:
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')  # xxx/test/step_xxx.csv --> step_xxx_tmp_rank
        for file_name in os.listdir(args.rcd_dir):
            if prefix in file_name:
                # load list of results
                with open(f'{args.rcd_dir}/{file_name}', 'rb') as f:
                    tmp = pickle.load(f)    
                for line in tmp:    # each line : [dataset_name, modality, sample_id, scores_of_labels(dict), label_names] 
                    evaluated_samples.add(f'{line[0]}_{line[2]}')
                    
    # dataset and loader
    # WARNING: Need to preprocess nii files to npy files, check https://github.com/zhaoziheng/SAT-DS/tree/main
    if args.online_crop:
        testset = Evaluate_Dataset_OnlineCrop(args.datasets_jsonl, args.max_queries, args.batchsize_3d, args.crop_size, evaluated_samples)
    else:
        testset = Evaluate_Dataset(args.datasets_jsonl, args.max_queries, args.batchsize_3d, args.crop_size, evaluated_samples)
        
    # # WARNING: Use this if you want to load the nii file directly
    # from data.nii_loader import Loader_Wrapper
    # from data.nii_evaluate_dataset import Evaluate_Dataset_OnlineCrop
    # testset = Evaluate_Dataset_OnlineCrop(
    #     jsonl_file=args.datasets_jsonl, 
    #     loader=Loader_Wrapper(),
    #     patch_size=args.crop_size,    # h w d
    #     max_queries=args.max_queries,
    #     batch_size=args.batchsize_3d,
    #     evaluated_samples=evaluated_samples
    # )
        
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=sampler, batch_size=1, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_fn, shuffle=False)
    sampler.set_epoch(0)
    
    # set model (by default gpu
    model = build_maskformer(args, device, gpu_id)
    
    # load knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=12,
        open_modality_embed=False,
        gpu_id=gpu_id,
        device=device
    )
    
    # load checkpoint if specified
    model, _, _ = load_checkpoint(
        checkpoint=args.checkpoint,
        resume=False,
        partial_load=args.partial_load,
        model=model, 
        device=device
    )
    
    # choose how to evaluate the checkpoint
    evaluate(model=model,
             text_encoder=text_encoder,
             device=device,
             testset=testset,
             testloader=testloader,
             csv_path=csv_path,
             resume=args.resume,
             save_interval=args.save_interval,
             dice_score=args.dice,
             nsd_score=args.nsd,
             visualization=args.visualization)

if __name__ == '__main__':
    # get configs
    args = parse_args()
    
    main(args)

    
    
    
        
    
    