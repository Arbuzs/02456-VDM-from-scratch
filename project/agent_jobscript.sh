#!/bin/bash
#BSUB -J WandB-Agent[1-4]  # <--- JOB ARRAY: Creates 4 separate jobs
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

# CPU cores and memory per agent
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"

# Wall time: Give agents enough time to train multiple models
# If one run takes 10 mins, 4 hours allows one agent to try ~20 configs
#BSUB -W 0:30

# Output files ( %J is JobID, %I is Array Index like 1, 2, 3, 4)
#BSUB -o Agent_%J_%I.out
#BSUB -e Agent_%J_%I.err

echo "=========================================="
echo "Agent ID: $LSB_JOBINDEX"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "=========================================="

# 1. Setup Environment
source $HOME/.bashrc
conda activate vdm_env

# 2. Run the Agent
# This command connects to WandB, pulls a set of hyperparameters,
# and runs your 'main.py' (via the 'program' entry in sweep_config.yaml).
# --count 10 means each agent will try 10 different models before stopping.
echo "Starting WandB Agent..."

# CORRECT (Based on your logs):
wandb agent --count 2 max-stalzer-danmarks-tekniske-universitet-dtu/VDM-from-scratch/30ts8815