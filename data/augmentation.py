import monai
from monai.transforms import (
    # Compose,
    RandShiftIntensityd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandScaleIntensityd,
    #RandScaleIntensityFixedMeand,
    #RandSimulateLowResolutiond,
    RandAdjustContrastd
)

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform

def get_SAT_augmentator(dataset_config, datasets):
    
    augmentator = {}

    # data augmentation (tailor for each dataset)
    for dataset in datasets:
        config = dataset_config[dataset]['augmentation']
        aug_ls = []
        if 'RandZoom' in config:
            aug_ls.append(RandZoomd(
                    keys=["image", "label"], 
                    mode=['area', 'nearest'],
                    min_zoom=config['RandZoom']['min_zoom'],
                    max_zoom=config['RandZoom']['max_zoom'],
                    prob=config['RandZoom']['prob'],
                )
            )
        if 'RandGaussianNoise' in config:
            aug_ls.append(
                RandGaussianNoised(
                    keys=['image'],
                    prob=config['RandGaussianNoise']['prob'],
                    mean=config['RandGaussianNoise']['mean'],
                    std=0.1
                )
            )
        if 'RandGaussianSharpen' in config:
            aug_ls.append(
                RandGaussianSharpend(
                    keys=['image'],
                    prob=config['RandGaussianSharpen']['prob'],
                )
            )
        if 'RandScaleIntensity' in config:
            aug_ls.append(
                RandScaleIntensityd(
                    keys=['image'],
                    factors=config['RandScaleIntensity']['factors'],
                    prob=config['RandScaleIntensity']['prob']
                )
            )
        """if 'RandScaleIntensityFixedMean' in config:
            aug_ls.append(
                RandScaleIntensityFixedMeand(
                    keys=['image'],
                    factors=config['RandScaleIntensityFixedMean']['factors'],
                    prob=config['RandScaleIntensityFixedMean']['prob']
                )
            )
        if 'RandSimulateLowResolution' in config:
            aug_ls.append(
                RandSimulateLowResolutiond(
                    keys=['image'],
                    prob=config['RandSimulateLowResolution']['prob']
                )
            )"""
        if 'RandAdjustContrastInvert' in config:
            aug_ls.append(
                RandAdjustContrastd(
                    keys=['image'],
                    #retain_stats=config['RandAdjustContrastInvert']['retain_stats'],
                    #invert_image=config['RandAdjustContrastInvert']['invert_image'],
                    gamma=config['RandAdjustContrastInvert']['gamma'],
                    prob=config['RandAdjustContrastInvert']['prob']
                )
            )
        if 'RandAdjustContrast' in config:
            aug_ls.append(
                RandAdjustContrastd(
                    keys=['image'],
                    #retain_stats=config['RandAdjustContrast']['retain_stats'],
                    #invert_image=config['RandAdjustContrast']['invert_image'],
                    gamma=config['RandAdjustContrast']['gamma'],
                    prob=config['RandAdjustContrast']['prob']
                )
            )
        if len(aug_ls) > 0:
            augmentator[dataset] = monai.transforms.Compose(aug_ls)

    return augmentator

def get_nnUNet_augmentator(datasets, xy_plane_size):
    
    tr_transforms = []
    
    tr_transforms.append(Convert3DTo2DTransform())
    
    tr_transforms.append(
        SpatialTransform(
            [xy_plane_size, xy_plane_size], patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=(-3.141592653589793, 3.141592653589793), angle_y=(0, 0), angle_z=(0, 0),
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=3,
            border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        )
    )
    
    # tr_transforms.append(SpatialTransform(
    #         patch_size_spatial, patch_center_dist_from_border=None,
    #         do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    #         do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
    #         p_rot_per_axis=1,  # todo experiment with this
    #         do_scale=True, scale=(0.7, 1.4),
    #         border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
    #         border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
    #         random_crop=False,  # random cropping is part of our dataloaders
    #         p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    #         independent_scale_for_each_axis=False  # todo experiment with this
    #     )
    # )
    
    tr_transforms.append(Convert2DTo3DTransform())
    
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                p_per_channel=0.5))
    
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=(0,)))  # ignore_axes
    
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
    
    tr_transforms.append(MirrorTransform((0, 1, 2)))    # tr_transforms.append(MirrorTransform(mirror_axes))
    
    tr_transforms.append(RemoveLabelTransform(-1, 0))
    
    tr_transforms.append(RenameTransform('seg', 'target', True))
    
    # tr_transforms.append(DownsampleSegForDSTransform2(
    #     [[1.0, 1.0, 1.0], [1.0, 0.5, 0.5], [0.5, 0.25, 0.25], [0.25, 0.125, 0.125], [0.125, 0.0625, 0.0625]], 
    #     0, input_key='target', output_key='target'))
    
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    
    tr_transforms = Compose(tr_transforms)
    
    augmentator = {}
    for dataset in datasets:
        augmentator[dataset] = tr_transforms
             
    return augmentator