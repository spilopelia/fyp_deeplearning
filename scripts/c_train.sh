#!/bin/bash -l

#SBATCH -p c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --output=logs3/d3m-%j.log
#SBATCH --time=3-00:00:00
#SBATCH -J "d3m"


export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export TORCH_HOME=/tmp/torch_cache

# enable logging    
export CUDA_LAUNCH_BLOCKING=1.

conda activate d3m
cd $SLURM_SUBMIT_DIR
srun python3 /home/user/ckwan1/ml/trainer.py $@ --gpus 4 --num_nodes 1 --num_workers ${SLURM_CPUS_PER_GPU}