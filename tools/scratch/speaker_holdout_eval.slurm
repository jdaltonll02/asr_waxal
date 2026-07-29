#!/bin/bash
#SBATCH --job-name=speaker_holdout_eval
#SBATCH --output=/home/jgibson2/projects/google_waxal/logs/speaker_holdout_eval_%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /home/jgibson2/projects/google_waxal/venv/bin/activate
cd /home/jgibson2/projects/google_waxal
export PYTHONPATH=/home/jgibson2/projects/google_waxal
export HF_HOME=/data/hf_cache/jgibson2/huggingface
python tools/scratch/speaker_holdout_eval.py
