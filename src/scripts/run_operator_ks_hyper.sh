#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
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

# Trains one hyper-nu FNO1D_hyper operator model per subset
# (pair/trio/quad/all, cf. config_command_operator_ks_hyper.yaml), for
# direct comparison against the per-nu-independent + linear-regression
# approach (run_operator_ks.sh + evaluate_operator_ensemble.py). The "all"
# subset (13 nu pooled) dominates the runtime; if the job hits the 24h
# wall, min_test_loss.pth from whichever subsets finished is still usable
# (evaluate_operator_ensemble.py falls back to it, and skips a subset
# whose checkpoint is missing entirely).
python entrypoints/main_operator_ks_hyper.py
source deactivate
