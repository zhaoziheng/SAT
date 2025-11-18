import torch
import numpy as np
import time
from medpy import metric
from .SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance

def calculate_metric_percase(pred, gt, dice=True, nsd=True):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
          
    metrics = {}
          
    if np.sum(gt) == 0.0:
        if np.sum(pred) == 0.0:
            if dice:
                metrics['dice'] = 1.0
            if nsd:
                metrics['nsd'] = 1.0
        else:
            if dice:
                metrics['dice'] = 0.0
            if nsd:
                metrics['nsd'] = 0.0
        return metrics
        
    if dice:
        dice_score = metric.binary.dc(pred, gt)
        metrics['dice'] = dice_score
    
    if nsd:
        surface_distances = compute_surface_distances(gt, pred, [1, 1, 3])
        nsd_score = compute_surface_dice_at_tolerance(surface_distances, 1)
        metrics['nsd'] = nsd_score
    
    return metrics

if __name__ == '__main__':
    pred = torch.zeros((3, 256, 256, 16)).numpy()
    pred[:, 0:128, 0:128, :] = 1.0
    gt = torch.zeros((3, 256, 256, 16)).numpy()
    gt[:, 0:64, 0:64, :] = 1.0
    dice = calculate_metric_percase(pred, gt)['dice']
    print(dice)
    
    
    