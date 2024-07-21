# SAT
This is the official repository for "One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompts" 🚀

[ArXiv](https://arxiv.org/abs/2312.17183)

[Website](https://zhaoziheng.github.io/SAT/)

[Model](https://huggingface.co/zzh99/SAT)

## Latest News:
🎉 We have released the code to build **SAT-DS**, a collection of 72 public segmentation datasets, contains over 22K 3D images, 302K segmentation masks and 497 classes from 3 different modalities (MRI, CT, PET) and 8 human body regions, upon which we build SAT. Check this [repo](https://github.com/zhaoziheng/SAT-DS/tree/main).

📢 We train a new version of SAT with larger model size (**SAT-Pro**) and more datasets (**72**), and it supports **497** classes now! 
We also renew SAT-Nano, and release some variants of SAT-Nano, based on different visual backbones ([U-Mamba](https://github.com/bowang-lab/U-Mamba/tree/main) and [SwinUNETR](https://arxiv.org/abs/2201.01266)) and text encoders ([MedCPT](https://huggingface.co/ncbi/MedCPT-Query-Encoder) and [BERT-Base](https://huggingface.co/google-bert/bert-base-uncased)). 
For more details about this update, refer to our new [paper](https://arxiv.org/abs/2312.17183).

⚠️ NOTE: We made lots of changes in this update, checkpoint/code from previous version are not compatible with the newly released code/checkpoint. However, the data format is consistent with before, so no need to re-prepare your data.

## Requirements
The implementation of U-Net relies on a customized version of [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures), to install it:
```
cd model
pip install -e dynamic-network-architectures-main
```

Some other key requirements:
```
torch>=1.10.0
numpy==1.21.5
monai==1.1.0 
transformers==4.21.3
nibabel==4.0.2
einops==0.6.1
positional_encodings==6.0.1
```

You also need to install `mamba_ssm` if you want the U-Mamba variant of SAT-Nano

## Inference Guidance (Command Line):
- S1. Build the environment following `requirements.txt`.

- S2. Download checkpoint of SAT and Text Encoder from [huggingface](https://huggingface.co/zzh99/SAT).
  
- S3. Prepare the data in a jsonl file. Check the demo in `data/inference_demo/demo.jsonl`.
    1. `image`(path to image), `labe`(name of segmentation targets), `dataset`(which dataset the sample belongs to) and `modality`(ct, mri or pet) are needed for each sample to segment. Modalities and classes that SAT supports can be found in in Table 12 of the paper.

    2. `orientation_code`(orientation) is `RAS` by default, which suits most images in axial plane. For images in sagittal plane (for instance, spine examination), set this to `ASR`.
The input image should be with shape `H,W,D` Our data process code will normalize the input image in terms of orientation, intensity, spacing and so on. Two successfully processed images can be found in `demo\processed_data`, make sure the normalization is done correctly to guarantee the performance of SAT.

- S4. Start the inference with SAT-Pro 🕶:
    ```
    torchrun \
    --nproc_per_node=1 \
    --master_port 1234 \
    inference.py \
    --rcd_dir 'demo/inference_demo/results' \
    --datasets_jsonl 'demo/inference_demo/demo.jsonl' \
    --vision_backbone 'UNET-L' \
    --checkpoint 'path to SAT-Pro checkpoint' \    
    --text_encoder 'ours' \
    --text_encoder_checkpoint 'path to Text encoder checkpoint' \
    --max_queries 256 \
    --batchsize_3d 2
    ```
    ⚠️ NOTE: `--batchsize_3d` is the batch size of input image patches, and need to be adjusted based on the gpu memory (check the table below);
    `--max_queries` is recommended to set larger than the classes in the inference dataset, unless your gpu memory is very limited;
    | Model | batchsize_3d | GPU Memory |
    |---|---|---|
    | SAT-Pro | 1 | ~ 34GB |
    | SAT-Pro | 2 | ~ 62GB |
    | SAT-Nano | 1 | ~ 24GB |
    | SAT-Nano | 2 | ~ 36GB |

- S5. Check `--rcd_dir` for outputs. Results are organized by datasets. For each case, the input image, aggregated segmentation result and a folder containing segmentations of each class will be found. All outputs are stored as nifiti files. You can visualize them using the [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).
  
- If you want to use SAT-Nano trained on 72 datasets, just modify `--vision_backbone` to 'UNET', and change the `--checkpoint` and `--text_encoder_checkpoint` accordingly.
  
- For other SAT-Nano variants (trained on 49 datasets):
  
  UNET-Ours: set `--vision_backbone 'UNET'` and `--text_encoder 'ours'`;

  UNET-CPT: set `--vision_backbone 'UNET'` and `--text_encoder 'medcpt'`;

  UNET-BB: set `--vision_backbone 'UNET'` and `--text_encoder 'basebert'`;

  UMamba-CPT: set `--vision_backbone 'UMamba'` and `--text_encoder 'medcpt'`;

  SwinUNETR-CPT: set `--vision_backbone 'SwinUNETR'` and `--text_encoder 'medcpt'`;

## Train Guidance:
Some preparation before start the training:
  1. you need to build your training data following this [repo](https://github.com/zhaoziheng/SAT-DS/tree/main), a jsonl containing all the training samples is required.
  2. you need to fetch the text encoder checkpoint from https://huggingface.co/zzh99/SAT to generate prompts.
Our recommendation for training SAT-Nano is 8 or more A100-80G, for SAT-Pro is 16 or more A100-80G. Please use the slurm script in `sh/` to start the training process. Take SAT-Pro for example:
  ```
  sbatch sh/train_sat_pro.sh
  ```



## TODO
- [ ] Inference demo on website.
- [x] Release the data preprocess code to build SAT-DS.
- [x] Release the train guidance.

## Citation
If you use this code for your research or project, please cite:
```
@arxiv{zhao2023model,
  title={One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompt}, 
  author={Ziheng Zhao and Yao Zhang and Chaoyi Wu and Xiaoman Zhang and Ya Zhang and Yanfeng Wang and Weidi Xie},
  year={2023},
  journal={arXiv preprint arXiv:2312.17183},
}
```
