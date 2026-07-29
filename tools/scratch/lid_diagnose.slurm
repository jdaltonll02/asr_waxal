#!/bin/bash
#SBATCH --job-name=lid_diagnose
#SBATCH --output=/home/jgibson2/projects/google_waxal/logs/lid_diagnose_%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

source /home/jgibson2/projects/google_waxal/venv/bin/activate
cd /home/jgibson2/projects/google_waxal
export HF_HOME=/data/hf_cache/jgibson2/huggingface
python tools/scratch/lid_diagnose.py
