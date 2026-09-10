#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH -J ks_operator_hyper_emul
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/KS_equation/ $SCRATCH/data/KS_equation

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Trains 3 hyper-nu FNO1D_hyper operator models, one per subset
# (pair/trio/all, cf. config_command_operator_ks_hyper.yaml), for direct
# comparison against the per-nu-independent + linear-regression approach
# (run_operator_ks.sh + evaluate_operator_ensemble.py).
python entrypoints/main_operator_ks_hyper.py
source deactivate
