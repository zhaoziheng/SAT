import datetime
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

from .dist import is_master

def set_up_log(args):
    # set exp time
    SHA_TZ = datetime.timezone(datetime.timedelta(hours=8), name='Asia/Shanghai')   
    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
    exp_time = f'{beijing_now.year}-{beijing_now.month}-{beijing_now.day}-{beijing_now.hour}-{beijing_now.minute}'
    
    # set exp name
    if args.name is None:
        args.name = exp_time
        
    # step up log path
    log_path = os.path.join(args.log_dir, args.name)
    if os.path.exists(log_path):
        print(
            f"Experiment {log_path} already exists."
        )
    os.makedirs(log_path, exist_ok=True)
    
    # record configs for reproducibility
    f_path = os.path.join(log_path, 'log.txt')
    with open(f_path, 'a') as f:
        configDict = args.__dict__
        f.write(f'{exp_time} \n Configs :\n')
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
    
    # Backup code for reproducibility
    # the file tree:
    # train.py, ...
    # train/trainer.py, loss.py ...
    # model/build_model.py, mask_former.py ...
    # ...
    # code_base = '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation'
    # code_backup = os.path.join(log_path, 'code')
    # os.makedirs(code_backup, exist_ok=True)
    # # find
    # for root, dirs, files in os.walk(code_base):
    #     for file in files:
    #         if file.endswith('.py') and 'log' not in root and 'deprecated' not in root:
    #             src_path = os.path.join(root, file)
    #             relative_path = os.path.relpath(src_path, code_base)
    #             target_path = os.path.join(code_backup, relative_path)
    #             os.makedirs(os.path.dirname(target_path), exist_ok=True)
    #             shutil.copy(src_path, target_path)
    #             print(f'{src_path} -> {target_path}')
                            
    # Backup Data（Jsonl） for reproducibility
    shutil.copy(args.datasets_jsonl, log_path)
    
    # Backup Dataset Config（json） for reproducibility
    shutil.copy(args.dataset_config, log_path)
    
    # checkpoint path
    checkpoint_dir = os.path.join(log_path, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # tensorboard path
    tensorboard_path = os.path.join(log_path, 'tensorboard_log')
    tb_writer = SummaryWriter(tensorboard_path)
    
    return checkpoint_dir, tb_writer, f_path