#!/bin/bash
#BSUB -J VDM-From-Scratch
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

# Email notifications
#BSUB -N

# CPU cores and memory
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"

# Max wall clock time (increased for training)
#BSUB -W 6:00

# Output files
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

# Print job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Start Time: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# --- FIX: ENSURE CONDA IS INITIALIZED AND ENVIRONMENT IS ACTIVATED ---

# Source the main bash configuration to ensure 'conda' command is available.
# The exact path may vary (e.g., ~/.bashrc or ~/.zshrc)
source $HOME/.bashrc 

# Activate the Conda environment
conda activate vdm_env

# --- RUN SCRIPT ---

# Print Python environment info (Use 'which' to confirm python is from vdm_env)
echo "Python path:"
which python

# Run the main script using the Python interpreter from the activated environment.
echo "Starting training..."
python project/main.py

# --- CLEANUP ---

# Capture exit code from the python script
EXIT_CODE=$?

# Deactivate environment
conda deactivate

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Exit with the same code as the main script
exit $EXIT_CODE