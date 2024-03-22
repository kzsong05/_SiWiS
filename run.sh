#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --job-name Amsen
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=350G
#SBATCH --output=slurm-%j.log

########## Command Lines to Run ##########

MASTER_ADDRESS="127.0.0.1"
MASTER_PORT="12397"

export CONDA3PATH=~/miniconda3/bin
module load Conda/3
conda activate torch

srun torchrun \
   --nnodes=1 \
   --nproc_per_node=8 \
   --max_restarts=0 \
   --rdzv_id=1 \
   --rdzv_backend=c10d \
   --rdzv_endpoint="${MASTER_ADDRESS}:${MASTER_PORT}" \
   main.py
