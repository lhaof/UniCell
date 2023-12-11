# UniCell
UniCell: Universal Cell Nucleus Classification via Prompt Learning, AAAI 2024

## Requisities
-`python=3.8`
-`pytorch=1.12.0`
-`torchvision=0.13.0`

## Installation
**step 0.** Install mmcv and mmengine using mim
```
pip install -U openmim
mim install mmengine==0.7.2
mim install mmcv==2.0.0
pip install timm==0.6.13
pip install ftfy==6.1.1
pip install regex==2023.3.23
pip install einops==0.6.0
```
**step 1.** Git clone the repository
```
git clone https://github.com/lhaof/UniCell.git
```
**step 2.** Install UniCell
```
cd UniCell/mmdetection
python -m pip install -e .
```
**step 3.** Install SAHI
```
cd projects/UniCell/tools/sahi
python -m pip install -e .
```

## Dataset Preparation
**step 0.** Download the dataset from [google drive](https) or [baidu drive](https) and unzip it to `UniCell/dataset/`.

**step 1.** Modify the `dataset_path` in `UniCell/tools/prepare_fourdataset_4Dataset_CMOL.py` and run it to generate the dataset.
```
python prepare_fourdataset_4Dataset_CMOL.py
```
## Training

## Testing
Modify the `path_to_dataset` and `checkpoint` path in `UniCell/tools/inference_multihead.py` and run it to test the model.
```
python inference_multihead.py
```