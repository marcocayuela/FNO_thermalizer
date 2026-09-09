#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J ks_operator_emul
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

# Whole KS_equation/ tree (all nu<X>/ subdirs) -- same data already used by
# run_emul_ks_hyper.sh, no new generation needed.
rsync -av $STORE/data/KS_equation/ $SCRATCH/data/KS_equation

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Trains 13 independent FNO1D operator-identification models (one per nu),
# one after another in a single job -- each is small/fast (modes=16,
# width=16, full spatial resolution but only ~1600 snapshots per nu).
python entrypoints/main_operator_ks.py
source deactivate
