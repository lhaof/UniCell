#!/bin/bash
#SBATCH -J UniCell
#SBATCH -N 1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -A p00120210008
#SBATCH -o UniCell.out
#SBATCH -p p-A100

module load gcc6/6.5.0
module load cuda11.3/toolkit/11.3.0

cd /home/huangjunjia/NewTopic_PromptDet/UniCell/mmdetection

python tools/train.py projects/UniCell/configs/UniCell_CMOL.py\
	--work-dir=/mntnfs/med_data2/huangjunjia/dataset/PromptDet/Checkpoints/Fourdataset_CMOL/GIT
