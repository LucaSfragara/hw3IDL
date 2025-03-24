#!/bin/bash
#SBATCH --job-name=face_verification    # a descriptive name for your job
#SBATCH --partition=GPU-shared          # request the GPU-shared partition
#SBATCH --gres=gpu:h100-80:1           
#SBATCH --time=6:00:00                 
#SBATCH --output=job_output_%j.txt     # output file with job id in its name
#SBATCH --error=job_error_%j.txt       # error file with job id in its name

# Change to your project directory
cd /local
module load anaconda3
conda activate env
export WANDB_DIR = "/tmp"
kaggle competitions download -c hw-3-p-2-automatic-speech-recognition-asr-11-785
unzip -q hw-3-p-2-automatic-speech-recognition-asr-11-785.zip 
cd /ocean/projects/cis250019p/sfragara/hw3

python main.py