#!/bin/bash
#SBATCH -J example
#SBATCH --gres=gpu:4
#SBATCH --output=example.out
#SBATCH --time 0-23:00:00
eval "$(conda shell.bash hook)"
conda activate tutorial
python main.py
