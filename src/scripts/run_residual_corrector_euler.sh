#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH -J euler_residual_corrector
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

# Needs generate_euler_correction_data.sh already run (and its output
# copied to $STORE, then staged here like the raw Euler data below).

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

mkdir -p $SCRATCH/data/euler_multi_quadrants_openBC/correction_data
rsync -av $STORE/data/euler_multi_quadrants_openBC/correction_data/ $SCRATCH/data/euler_multi_quadrants_openBC/correction_data/

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_residual_corrector_euler.py
source deactivate
