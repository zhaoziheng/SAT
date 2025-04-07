import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .cvpr25_train_dataset import Med_SAM_Dataset
from .collect_fn import collect_fn


def build_dataset(args):
    dataset = Med_SAM_Dataset(
        jsonl_file=args.datasets_jsonl,
        text_prompts_json=args.text_prompts_json,
        crop_size=args.crop_size,    # h w d
        max_queries=args.max_queries,
        dataset_config=args.dataset_config,
        allow_repeat=args.allow_repeat,
        nnUNet_aug = args.nnUNet_aug
    )
    sampler = DistributedSampler(dataset)
    if args.num_workers is not None:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory)
    
    return dataset, dataloader, sampler