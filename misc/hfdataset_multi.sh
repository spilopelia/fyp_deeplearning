#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --output=logs3/hf-%j.log
#SBATCH --time=7-00:00:00
#SBATCH -J "hf"

conda activate d3m
cd $SLURM_SUBMIT_DIR
srun python /home/user/ckwan1/ml/misc/hfdataset_multi.py 