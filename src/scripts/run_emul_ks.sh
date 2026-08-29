#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J ks_nu0p35_emul
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/KS_equation/ $SCRATCH/data/KS_equation

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_emul_ks.py
source deactivate
