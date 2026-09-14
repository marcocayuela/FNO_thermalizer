#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=3:00:00
#SBATCH -J shear_flow_Re5e4_Sc1e0_diff
#SBATCH -o /scratch/cayuelam/logs/shear_flow/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/shear_flow/ $SCRATCH/data/shear_flow

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_diffusion_shear_flow.py
source deactivate
