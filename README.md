# SAT
This is the official repository for "One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompts"

[ArXiv](https://arxiv.org/abs/2312.17183)
[Website](https://zhaoziheng.github.io/SAT/)

## Latest News:
The inference code and model checkpoint of SAT-Nano, SAT-Pro and all the variants mentioned in the paper have been released.

## Inference Guidance:
- S1. Build the environment following the requirements.txt (you also need `mamba_ssm` if you want the U-Mamba variant of SAT-Nano).
- S2. Download checkpoint of SAT and Text Encoder from [huggingface](https://huggingface.co/zzh99/SAT) or [baiduyun](https://pan.baidu.com/s/1WJdsCAHYgfOxXwU1RyfUcg?pwd=kjjy) (passcode:kjjy).
- S3. Prepare the data to inference in a jsonl file. A demo can be found in `data/inference_demo/demo.jsonl`. Image path, label name, dataset name and modality (ct, mri or pet) are needed for each sample to segment.
Also, orientation is critical for consistent performance. The orientation code is `RAS` by default, which suits most images in axial plane.
Modalities and classes that SAT supports can be found in in Table 12 of the paper.
Download links of the dataset involved in SAT-DS can be found in Table 13 of the paper.
- S4. Start the inference with SAT-Pro:
    ```
    torchrun \
    --nproc_per_node=1 \
    --master_port 1234 \
    inference.py \
    --rcd_dir 'demo/inference_demo/results'
    --data_jsonl 'demo/inference_demo/demo.jsonl' \
    --vision_backbone 'UNET-L' \
    --checkpoint 'path to SAT-Pro checkpoint' \
    --text_encoder 'ours' \
    --text_encoder_checkpoint 'path to Text encoder checkpoint' \
    --max_queries 256 \
    --batchsize_3d 2
    ```
    It's recommended to set `--max_queries` larger than the classes in the inference dataset, and modify `--batchsize_3d` based on the computation resource. If the gpu memory is limited, `--max_queries` can be set smaller, at the cost of inference speed.
- S5. Check `--rcd_dir` where the outputs are generated. Results are organized by the dataset. For each case, the input image, aggregated segmentation result and a folder containing segmentations of each class will be found. All outputs are stored as nifiti files. You can visualize them using the [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php)."

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

# Citation
If you use this code for your research or project, please cite:
```
@arxiv{zhao2023model,
  title={One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompt}, 
  author={Ziheng Zhao and Yao Zhang and Chaoyi Wu and Xiaoman Zhang and Ya Zhang and Yanfeng Wang and Weidi Xie},
  year={2023},
  journal={arXiv preprint arXiv:2312.17183},
}
```
