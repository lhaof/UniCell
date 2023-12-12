# UniCell
<img src=./resources/intro.jpg width=30%/>

UniCell: Universal Cell Nucleus Classification via Prompt Learning, AAAI 2024

## Overall Framework
![](./resources/overall_framework.jpg)

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
**step 0.** Download the dataset from [google drive](https://drive.google.com/file/d/1VFQnAvVM918pS3co7J_hbGcGsOvJVuo1/view?usp=sharing) and unzip it to `UniCell/dataset/`.

**step 1.** Modify the `dataset_path` in `projects/UniCell/tools/prepare_fourdataset_4Dataset_CMOL.py` and run it to generate the dataset.
```
python prepare_fourdataset_4Dataset_CMOL.py
```

## Training
Modify the `dataset_path` in `projects/UniCell/configs/nuclei_det_multihead_cmol.py` and run the following command to train the model.
### Training on a single GPU
```shell
python ./tools/train.py tools/train.py projects/UniCell/configs/UniCell_CMOL.py\
	--work-dir=${SAVE_DIR}
```
### Training on multiple GPUs
```shell
bash ./tools/dist_train.sh projects/UniCell/configs/UniCell_CMOL.py\
    ${GPU_NUM} --work-dir=${SAVE_DIR}
```

## Testing
**step 0.** Download the universal model from [google drive](https://drive.google.com/file/d/1dLZj-R9cXhiRjvU2oKPsAM8qZN9079uZ/view?usp=sharing).

**step 1.** Modify the `path_to_dataset` and `checkpoint` path in `projects/UniCell/tools/inference_multihead.py` and run it to test the model.
```
python inference_multihead.py
```

We use the entire training set for training, and use the final model for testing after completing 160k iterations.


## Training your own datasets

**step 0.** Transfer your dataset into the [format](./resources/README.txt) of `HoverNet CoNSeP` and put it in the `UniCell/dataset/` directory.

**step 1.** Modify the `dataset_path`, `categories`, `datasets` in `projects/UniCell/tools/prepare_fourdataset_4Dataset_CMOL.py` and run it to generate the dataset.

Noted that the dataset name in `trans_to_patch` function should be the same as the dataset name in `datasets`. 

**step 2.** Modify the `METAINFO` in `projects/UniCell/configs/nuclei_det_multihead_cmol.py` and make sure that the `num_classes` is equal to the number of categories in your dataset.

(You can register a new dataset type instead if you are familiar with mmdetection.)

**step 3.** Modify the `num_classes` in `projects/UniCell/configs/UniCell_CMOL.py`.

**step 4.** Modify the `dataset_names` and `category_names` in `projects/UniCell/configs/UniCell_CMOL.py (text_cfg)`.

Modify the `mask_map`, the key corresponds to the dataset, and the value corresponds to the category that requires mask. 

**step 5.** Training. Follow the steps in [Training](#Training).


## Acknowledgement
We thank the following projects for their valuable contributions to this work.
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [SAHI](https://github.com/obss/sahi)