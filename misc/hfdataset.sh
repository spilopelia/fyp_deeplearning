#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --output=logs2/hf-%j.log
#SBATCH --time=7-00:00:00
#SBATCH -J "hf"

conda activate d3m
cd $SLURM_SUBMIT_DIR
python3 /home/user/ckwan1/ml/hfdataset.py 