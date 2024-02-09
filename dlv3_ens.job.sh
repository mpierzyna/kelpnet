#!/bin/bash
#####
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=64   # This needs to match Trainer(devices=...)
#SBATCH --mem=0

# Set variables
CONDA_ENV=/home/max/mambaforge/envs/keras3

# Load conda environment
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

python -u torch_dlv3_ens.py