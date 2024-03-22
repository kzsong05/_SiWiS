#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --job-name wifi
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.log

########## Command Lines to Run ##########
export CONDA3PATH=~/miniconda3/bin
module load Conda/3
conda activate torch

srun python process.py
