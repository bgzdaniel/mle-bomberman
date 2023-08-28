#!/bin/zsh

#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=rl_output_cnnfc.txt
#SBATCH --nodelist=ceg-brook[02]

module load anaconda/3
source ~/.bashrc
conda activate newest_torch

python3 main.py play --no-gui --agents cnnfc_agent peaceful_agent peaceful_agent peaceful_agent --train 1 --scenario coin-trainer --n-rounds 10000

