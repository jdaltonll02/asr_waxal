#!/bin/bash
#SBATCH --job-name=lm_lid_validate
#SBATCH --output=/home/jgibson2/projects/google_waxal/logs/lm_lid_validate_%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

source /home/jgibson2/projects/google_waxal/venv/bin/activate
cd /home/jgibson2/projects/google_waxal
export PYTHONPATH=/home/jgibson2/projects/google_waxal
export HF_HOME=/data/hf_cache/jgibson2/huggingface
python tools/scratch/lm_lid_validate.py
