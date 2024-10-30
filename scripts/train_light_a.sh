#!/bin/bash -l

#SBATCH -p a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=50G
#SBATCH --output=logs/d3m-%j.log
#SBATCH --time=3-00:00:00
#SBATCH -J "d3m"


export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

# enable logging    
export CUDA_LAUNCH_BLOCKING=1.

conda activate d3m
cd $SLURM_SUBMIT_DIR
srun python3 /home/user/ckwan1/ml/trainer.py $@ 