#!/bin/bash -l

#SBATCH -p a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --output=logs3/d3m-%j.log
#SBATCH --time=3-00:00:00
#SBATCH -J "d3m"


export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export TORCH_HOME=/tmp/torch_cache
export NCCL_DEBUG=TRACE
# enable logging    
export CUDA_LAUNCH_BLOCKING=1.

conda activate d3m
cd $SLURM_SUBMIT_DIR
srun python3 /home/user/ckwan1/ml/trainer.py $@ --gpus 2 --num_nodes 1 --num_workers ${SLURM_CPUS_PER_GPU}