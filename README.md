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
We have trained the a baseline model (SAT-Nano) on 10 percent training data. The checkpoints can be found in [huggingface](https://huggingface.co/zzh99/SAT/tree/main/Nano).

## Train Guidance:
- **Data preparation**. We preprocess and organize the training data in a jsonl file `data/challenge_data/train_10percent.jsonl`. For each sample, `data` refers to the npz file; `dataset` must be aligned with `data/challenge_data/CVPR25_TextSegFMData_with_class.json` to assign label and text prompts; `modality` is used to encode text prompts; `label_existance` is used for training data sampling and can be derived with `data/check_label.py`.
- **Knowledge enhancement**. You can either use our pretrained text encoder in [huggingface](https://huggingface.co/zzh99/SAT/tree/main/Pretrain) or re-do the pretraining with guidance in this [repo](https://github.com/zhaoziheng/SAT-Pretrain/tree/master). We suggest freezing the text encoder when training the segmentation model.
- **Segmentation**. The training script is in `sh/cvpr2025.sh`. The training take around 3 days with 4xA100-80GB. You can may modify the patch size and batch size to train on GPUs with less memory.

## Evaluation Guidance:
You need to prepare the validation data in a jsonl file `data/challenge_data/validation_subset.jsonl`. For each sample, `img_path`, `gt_path`, `dataset` and `modality` are required. Then, run the evaluation script in `sh/eval_cvpr2025.sh`.

## Inference Guidance:
We provide inference code for testing data:
```
python inference.py
```
This will read a sample fron `inputs/`, generate and same the prediction mask to `outputs/`. 
Alternatively, you can run our docker from [dropbox](https://www.dropbox.com/scl/fo/r616q5oaxgsq740txya88/ALjxrzUOUzB7KbPs0WDiS0s?rlkey=zmfxbd8xo9t9dlwju38nbokq8&st=wy1tpl0n&dl=0):
```
docker container run --gpus "device=0" -m 32G --name sat_cvpr25 --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ sat_cvpr25:latest /bin/bash -c "sh predict.sh"
```

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
