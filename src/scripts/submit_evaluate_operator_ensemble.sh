#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=02:00:00
#SBATCH -J ks_operator_ensemble_eval
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

# Usage: sbatch submit_evaluate_operator_ensemble.sh
# Needs run_operator_ks.sh already completed (all 13 nu checkpoints saved).

module purge
module load python/3.11

source activate fto

export DATA_DIR=/scratch/cayuelam/data/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python evaluation/evaluate_operator_ensemble.py \
    --data_dir "$DATA_DIR" \
    --exp_dir KS_equation \
    --run_dir /scratch/cayuelam/fno/runs/KS_equation \
    --exp_name fno_ks_operator \
    --out_dir /scratch/cayuelam/evaluate_operator_ensemble

echo "Rsync back to local recommended once done:"
echo "  rsync -av mesu:/scratch/cayuelam/evaluate_operator_ensemble/ /Users/marco/Documents/PhD/thermalizer/evaluate_operator_ensemble/"

source deactivate
