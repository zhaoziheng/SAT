import os
import time

import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from .dist import is_master
from .loss import segmentation_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1): # avg value over samples, and the num of samples, e.g. the avg loss over a batch and batchsize
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Trainer():
    def __init__(self,
                 args, 
                 model,
                 text_encoder,
                 device, 
                 trainset, 
                 trainloader, 
                 sampler, 
                 dice_loss, 
                 bce_w_logits_loss, 
                 optimizer, 
                 scheduler, 
                 tb_writer, 
                 checkpoint_dir, 
                 log_file):
        
        # average meters
        self.dataset_dice_loss_m_dict = {}
        self.dataset_ce_loss_m_dict = {}
        self.dice_loss_m = AverageMeter()
        self.ce_loss_m = AverageMeter()
        self.data_time_m = AverageMeter()
        self.query_time_m = AverageMeter()
        self.compute_time_m = AverageMeter()
        self.loss_time_m = AverageMeter()
        self.bp_time_m = AverageMeter()
        self.sample_statistics = {}

        # model
        self.model = model
        self.text_encoder = text_encoder
        self.device = device
        
        # data
        self.trainset = trainset
        self.trainloader = iter(trainloader)    
        self.sampler = sampler 
        
        # loss calculator, optimizer and lr
        self.dice_loss = dice_loss 
        self.bce_w_logits_loss = bce_w_logits_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # log
        self.tb_writer = tb_writer
        self.log_file = log_file
        self.total_steps = args.step_num
        if isinstance(self.total_steps, list):  # could be multi-stage restart lr
            self.total_steps = sum(self.total_steps)
        self.log_step_interval = args.log_step_interval
        if self.log_step_interval is None:
            self.log_step_interval = 1000
        
        # checkpoint   
        self.checkpoint_dir = checkpoint_dir
        self.save_large_interval = args.save_large_interval
        self.save_small_interval = args.save_small_interval
            
        # amp
        self.grad_scaler = GradScaler()
        
        # accumulate grad
        self.accumulation_steps = 0
        self.accumulate_grad_interval = args.accumulate_grad_interval
        
        self.model.train()
        
    def train_one_step(self, step):
        
        end_time = time.time()        

        # step-wise scheduling
        self.scheduler(step)
            
        # data loading
        batch = next(self.trainloader)
        text, modality, image, mask, dataset_name, query_mask = batch['text'], batch['modality'], batch['image'], batch['mask'], batch['dataset'], batch['query_mask']
        batch_size = len(text)
        image = image.to(device=self.device)
        mask = mask.to(device=self.device)
        query_mask = query_mask.to(device=self.device)   # B C
        
        if is_master():
            self.data_time_m.update(time.time()-end_time)
            end_time = time.time()
        
        with autocast():
            # get query embeddings from label name and image modality
            queries = self.text_encoder(text, modality)
            
            if is_master():
                self.query_time_m.update(time.time()-end_time)
                end_time = time.time()
        
            # forward
            logits = self.model(queries=queries, image_input=image)   # bnhwd or list of bnhwd (deep supervision)
            
            if is_master():
                self.compute_time_m.update(time.time()-end_time)
                end_time = time.time()
            
            # generate weight for each layer output
            weights = torch.Tensor([1 / (2 ** i) for i in range(len(logits))])  # 1, 1/2, 1/4, 1/8
            weights = weights / weights.sum()
            # loss of the largest output
            bp_loss, unreduced_batch_dice_loss, unreduced_batch_ce_loss = segmentation_loss(logits[0], mask, query_mask, self.dice_loss, self.bce_w_logits_loss, weights[0])
            # loss of mid-stage output
            for mid_logits, mid_weight in zip(logits[1:], weights[1:]):
                mid_mask = torch.nn.functional.interpolate(mask, size=mid_logits.shape[2:], mode='nearest')  # HWD -> hwd
                tmp_bp_loss, tmp_unreduced_batch_dice_loss, tmp_unreduced_batch_ce_loss = segmentation_loss(mid_logits, mid_mask, query_mask, self.dice_loss, self.bce_w_logits_loss, mid_weight)
                bp_loss += tmp_bp_loss
                unreduced_batch_dice_loss += tmp_unreduced_batch_dice_loss
                unreduced_batch_ce_loss += tmp_unreduced_batch_ce_loss
        
        # scale the loss
        self.grad_scaler.scale(bp_loss).backward()
        self.accumulation_steps += 1
        
        if is_master():
            self.loss_time_m.update(time.time()-end_time)
            end_time = time.time()
        
        # accumulate grad or bp and update the model
        if self.accumulation_steps % self.accumulate_grad_interval == 0:
            self.grad_scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.grad_scaler.update()
            self.accumulation_steps = 0
        
        if is_master():
            self.bp_time_m.update(time.time()-end_time)
            end_time = time.time()

        with torch.no_grad():
            if is_master():
                # attribute loss to each dataset
                for i, name in enumerate(dataset_name):
                    if name not in self.dataset_dice_loss_m_dict:
                        self.dataset_dice_loss_m_dict[name] = AverageMeter()
                        self.dataset_ce_loss_m_dict[name] = AverageMeter()
                    self.dataset_dice_loss_m_dict[name].update(unreduced_batch_dice_loss[i].item(), 1)
                    self.dataset_ce_loss_m_dict[name].update(unreduced_batch_ce_loss[i].item(), 1)
                    if name not in self.sample_statistics:
                        self.sample_statistics[name] = 0
                    self.sample_statistics[name] += 1
                # overall loss
                self.dice_loss_m.update(torch.mean(unreduced_batch_dice_loss).item(), batch_size)
                self.ce_loss_m.update(torch.mean(unreduced_batch_ce_loss).item(), batch_size)
            
                # log and save regularly (and only after a update of the model in case of accmulation grad)
                if self.accumulation_steps == 0:
                    
                    # log regularly
                    if step % self.log_step_interval == 0:  
                        
                        # tensorboard
                        lr = self.optimizer.param_groups[0]['lr']
                        self.tb_writer.add_scalar('train_dice_loss/all_dataset', self.dice_loss_m.avg, step)
                        self.tb_writer.add_scalar('train_ce_loss/all_dataset', self.ce_loss_m.avg, step)
                        self.tb_writer.add_scalar('train/learning_rate', lr, step)
                        for name, meter in self.dataset_dice_loss_m_dict.items():
                            self.tb_writer.add_scalar(f'train_dice_loss/{name}', meter.avg, step)
                        for name, meter in self.dataset_ce_loss_m_dict.items():
                            self.tb_writer.add_scalar(f'train_ce_loss/{name}', meter.avg, step)
                        
                        # print log    
                        info = f"\nStep {step}({(step/self.total_steps):.3f}) | LR {lr:.4e} "
                        info += f"| Dice Loss {self.dice_loss_m.val:.4f}({self.dice_loss_m.avg:.4f}) | CE Loss {self.ce_loss_m.val:.4f}({self.ce_loss_m.avg:.4f}) |"
                        info += f"| Data Time {self.data_time_m.avg:.2f} | Generate Query Time {self.query_time_m.avg:.2f} | FW Time {self.compute_time_m.avg:.2f} |"
                        info += f"| Loss Time {self.loss_time_m.avg:.2f} | BP Time {self.bp_time_m.avg:.2f}\n"
                        print(info)
                        
                        # write log
                        with open(self.log_file, 'a') as f:
                            f.write(info)
                            # dice loss
                            f.write(f'| Dice Loss |')
                            sorted_keys = sorted(self.dataset_dice_loss_m_dict.keys())
                            for name in sorted_keys:
                                meter = self.dataset_dice_loss_m_dict[name]
                                f.write(f' {name} {meter.avg} |')
                            # bce loss
                            f.write(f'\n| BCE Loss |')
                            sorted_keys = sorted(self.dataset_ce_loss_m_dict.keys())
                            for name in sorted_keys:
                                meter = self.dataset_ce_loss_m_dict[name]
                                f.write(f' {name} {meter.avg} |')
                            f.write(f'\n')
                            # sample statistics
                            f.write(f'\n| Sample Statistics |')
                            sorted_keys = sorted(self.sample_statistics.keys())
                            for name in sorted_keys:
                                sampled_times = self.sample_statistics[name]
                                size, repeated_times = self.trainset.get_size_and_repeat(name)
                                f.write(f' {name} {sampled_times}/{size}/{size+repeated_times} |')
                            f.write(f'\n')
                            
                        # reset records
                        for name, meter in self.dataset_dice_loss_m_dict.items():
                            meter.reset()
                        for name, meter in self.dataset_ce_loss_m_dict.items():
                            meter.reset()
                        for k in self.sample_statistics.keys():
                            self.sample_statistics[k] = 0
                        self.dice_loss_m.reset()
                        self.ce_loss_m.reset()
                        self.data_time_m.reset()
                        self.query_time_m.reset()
                        self.compute_time_m.reset()
                        self.loss_time_m.reset()
                        self.bp_time_m.reset()
                   
                    # save regularly          
                    if step % self.save_large_interval == 0:
                        t0 = time.time()
                        torch.save({'step':step,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),         
                                    }, os.path.join(self.checkpoint_dir, f'step_{step}.pth'))
                        tmp_path = os.path.join(self.checkpoint_dir, f'text_encoder_step_{step}.pth')
                        torch.save({'step':step,
                                    'model_state_dict': self.text_encoder.model.state_dict(),
                                    }, tmp_path)
                        print(f'Save, time {time.time()-t0}s')
                    
                    # save more frequently to avoid interruption 
                    if self.save_small_interval and step % self.save_small_interval == 0:
                        t0 = time.time()
                        torch.save({'step': step,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),         
                                    }, os.path.join(self.checkpoint_dir, f'latest_step.pth'))
                        tmp_path = os.path.join(self.checkpoint_dir, f'text_encoder_latest_step.pth')
                        torch.save({'step':step,
                                    'model_state_dict': self.text_encoder.model.state_dict(),
                                    }, tmp_path)
                        print(f'Save, time {time.time()-t0}s')
                