#!/bin/zsh

#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --exclude=octane[001-008],ceg-victoria
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=output_cnn_agent_simpleDQN_classic.txt

module load anaconda/3
source ~/.bashrc
conda activate newest_torch

python3 main.py play --no-gui --agents cnn_agent_simpleDQN rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds 10000

