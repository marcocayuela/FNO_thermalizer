#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J kolmogorov34
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/kolmogorov/ $SCRATCH/data/kolmogorov

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_diffusion_unet.py

# Visualisation automatique : re-genere les courbes de tous les runs sous
# kolmogorov/Re90 (dont celui-ci) + une comparaison inter-runs par famille.
python evaluation/plot_training_metrics.py \
    --runs_base "$LOG_DIR/kolmogorov/Re90" \
    --compare_out "$LOG_DIR/kolmogorov/Re90/training_comparison.png" \
    --summary_out "$LOG_DIR/kolmogorov/Re90/training_summary.csv"

conda deactivate
