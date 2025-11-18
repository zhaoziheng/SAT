import numpy as np

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step / warmup_length)

def cosine_lr(optimizer, 
              base_lr:list, 
              warmup_length:list, 
              steps:list):
    """
    Restart Cosine Annealing: Split the training process into N subprocesses, each follows a cosine anealing lr with warmup
    e.g. 
    base_lr = [1e-4, 5e-5, 1e-5]
    warmup_length = [10K, 10K, 10K]
    steps = [50k, 50k, 50k]
    """
    if not isinstance(base_lr, list):
        base_lr = [base_lr]
    if not isinstance(base_lr, list):
        warmup_length = [warmup_length]
    if not isinstance(base_lr, list):
        steps = [steps]
    
    def _lr_adjuster(step):
        accumulate_steps = 0
        for current_steps, current_warmup_length, current_base_lr in zip(steps, warmup_length, base_lr):
            if accumulate_steps + current_steps >= step: # in this training subprocess
                break
            else:
                accumulate_steps += current_steps
                
        current_step = step - accumulate_steps
        
        if current_step <= current_warmup_length:
            lr = _warmup_lr(current_base_lr, current_warmup_length, current_step)
        else:
            e = current_step - current_warmup_length
            es = current_steps - current_warmup_length
            if e == es:  # otherwise for last step, lr=0
                es += 1
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * current_base_lr
            
        assign_learning_rate(optimizer, lr)
        return lr
    
    return _lr_adjuster