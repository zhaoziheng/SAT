# SAT for [CVPR 2025: FOUNDATION MODELS FOR TEXT-GUIDED 3D BIOMEDICAL IMAGE SEGMENTATION](https://www.codabench.org/competitions/5651/)

SAT is one of the official baselines in [CVPR 2025: FOUNDATION MODELS FOR TEXT-GUIDED 3D BIOMEDICAL IMAGE SEGMENTATION](https://www.codabench.org/competitions/5651/). This repo is a simple guidance to train and evalute SAT in the challenge.

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

## Model Checkpoints
We have trained the baseline model on 10 percent training data. The checkpoints can be found in [huggingface](https://huggingface.co/zzh99/SAT/tree/main/CVPR25).

## Data preparation
For convenient, we preprocess and organize the training data in a jsonl file `data/challenge_data/train_10percent.jsonl`. For each sample, `data` refers to the npz file; `dataset` must be aligned with `data/challenge_data/CVPR25_TextSegFMData_with_class.json` to assign label and text prompts; `modality` is used to encode text prompts; `label_existance` is used for training data sampling and can be derived with `data/check_label.py`.

## Inference Guidance:
TBD

## Train Guidance:
- **Knowledge enhancement**. You can either use our pretrained text encoder in [huggingface](https://huggingface.co/zzh99/SAT/tree/main/Pretrain) or re-do the pretraining with guidance in this [repo](https://github.com/zhaoziheng/SAT-Pretrain/tree/master).
- **Segmentation**. The training script is in `sh/cvpr2025.sh`.

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
