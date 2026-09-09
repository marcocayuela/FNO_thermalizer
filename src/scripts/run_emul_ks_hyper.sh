#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J ks_hyper_nu_emul
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

# Whole KS_equation/ tree (all nu<X>/ subdirs, cf.
# config_command_emul_ks_hyper.yaml's nu_values) -- same source/target as
# run_emul_ks.sh, just no longer a single nu.
rsync -av $STORE/data/KS_equation/ $SCRATCH/data/KS_equation

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_emul_ks_hyper.py
source deactivate
