import torch
import torch.nn.functional as F
from einops import repeat

def collect_fn(data):
    """
    Pad images and masks to the same depth and num of class
    
    Args:
        data : [{'text':..., 'image':..., 'mask':..., 'modality':..., 'image_path':..., 'mask_path':..., 'dataset':..., 'y1x1z1_y2x2z2':...}, ...]
    """
    
    image = []
    mask = []
    text = []
    modality = []
    image_path = []
    mask_path = []
    dataset = []
    y1x1z1_y2x2z2 = []

    # pad to max depth in the batch
    # pad to max num of class in the batch
    max_class = 1
    for sample in data:
        class_num = sample['mask'].shape[0]
        max_class = class_num if class_num > max_class else max_class
        
    query_mask = torch.zeros((len(data), max_class)) # bn
    for i, sample in enumerate(data):
        if sample['image'].shape[0] == 1:
            sample['image'] = repeat(sample['image'], 'c h w d -> (c r) h w d', r=3)
        image.append(sample['image'])
        
        class_num = sample['mask'].shape[0]
        pad = (0, 0, 0, 0, 0, 0, 0, max_class-class_num)
        padded_mask = F.pad(sample['mask'], pad, 'constant', 0)   # nhwd
        mask.append(padded_mask)
        sample['text'] += ['none'] * (max_class-class_num)
        query_mask[i, :class_num] = 1.0
        
        text.append(sample['text'])   
        modality.append(sample['modality'])
        image_path.append(sample['image_path'])
        mask_path.append(sample['mask_path'])
        dataset.append(sample['dataset'])
        y1x1z1_y2x2z2.append(sample['y1x1z1_y2x2z2'])
    
    image = torch.stack(image, dim=0)
    mask = torch.stack(mask, dim=0).float()
    return {'image':image, 'mask':mask, 'text':text, 'modality':modality, 'image_path':image_path, 'mask_path':mask_path, 'dataset':dataset, 'query_mask':query_mask, 'y1x1z1_y2x2z2':y1x1z1_y2x2z2}
