import argparse

def str2bool(v):
    return v.lower() in ('true', 't')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Exp ID
    
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the exp",
    )
    
    # MaskFormer
    
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help='UNETs or SwinUNETR or Mamba'
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs='+',
        help='patch size on h w and d'
    )
    parser.add_argument(
        "--deep_supervision",
        type=str2bool,
    )
    
    # Checkpoint
    
    parser.add_argument(
        "--resume",
        type=str2bool,
        help="Resume an interrupted exp",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--partial_load",
        type=str2bool,
        help="Allow to load partial paramters from checkpoint",
    )
    parser.add_argument(
        "--inherit_knowledge_encoder",  # inherit UNET Encoder from pretraining (not necessary)
        type=str2bool,
        default=False,
    )
    
    # Save and Log
    
    parser.add_argument(
        "--save_large_interval", 
        type=int, 
        help="Save checkpoint regularly"
    )
    parser.add_argument(
        "--save_small_interval", 
        type=int, 
        help="Save checkpoint more frequentlt in case of interruption (not necessary)"
    )
    parser.add_argument(
        "--log_step_interval", 
        type=int, 
        help="Output and record log info every N steps"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log',
        help="Dir of exp logs",
    )
    
    # Randomness
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    
    # GPU
    
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    
    # Loss function, Optimizer
    
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-8
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--accumulate_grad_interval", 
        type=int, 
        help="accumulate grad"
    )
    
    # Learing Rate 
    
    parser.add_argument(
        "--step_num",
        type=int,
        nargs='+',
        help="Total step num",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        nargs='+',
        help="Warm up step num",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs='+',
        help="Peak learning rate",
    )
    
    # Med SAM Dataset
    
    parser.add_argument(
        "--datasets_jsonl",
        type=str,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        help='path to dataset config parameters'
    )
    
    # Sampler and Loader
    
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs='+',
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        help="max queries for each sample",
    )
    parser.add_argument(
        "--batchsize_3d",
        type=int,
        help="batchsize of image",
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help='load data to gpu to accelerate'
    )
    parser.add_argument(
        "--allow_repeat",
        type=str2bool,
        help='repeat multipy times for sample with too many labels (to accelerate convergency)'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
    )
    
    # Text Encoder
    
    parser.add_argument(
        "--text_encoder_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        help="you have to choose your text encoder",
    )
    parser.add_argument(
        "--open_bert_layer",
        type=int,
        default=12,
        help="open bert layers ahead of this",
    )
    parser.add_argument(
        "--open_modality_embed",
        type=str2bool,
        default=True,
        help="open modality embed",
    )
    parser.add_argument(
        "--text_encoder_partial_load",
        type=str2bool,
        default=False,
        help="Allow to load partial paramters",
    )
    
    args = parser.parse_args()
    return args