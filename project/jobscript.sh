#!/bin/bash
#BSUB -J DLCV-part-2-project
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -N

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"

# Max wall clock time is 12 hours
#BSUB -W 1:00 

#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

source ~/02516-IDLCV/.venv/bin/activate
python train.py