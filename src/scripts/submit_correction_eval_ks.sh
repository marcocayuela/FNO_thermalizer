#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=04:00:00
#SBATCH -J correction_eval_ks
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

# Usage: sbatch submit_correction_eval_ks.sh [rollout_steps] [sim_file] [out_dir]
# Needs both run_emul_ks.sh and run_diff_ks.sh already completed.

module purge
module load python/3.11

source activate fto

mkdir -p /scratch/cayuelam/logs/kolmogorov/

export DATA_DIR=$SCRATCH/data/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

ROLLOUT=${1:-5000}
SIM_FILE=${2:-sim1.h5}
OUT_DIR=${3:-$SCRATCH/correction_eval_ks_nu0p35}

python evaluation/correction_eval_ks.py \
    --data_dir "$DATA_DIR" \
    --exp_dir KS_equation/nu0p35 \
    --sim_file "$SIM_FILE" \
    --emulator_run "$SCRATCH/fno/runs/KS_equation/nu0p35/fno_ks_nu0p35" \
    --diffusion_run "$SCRATCH/fno/runs/KS_equation/nu0p35/diffusion_ks_nu0p35" \
    --nu 0.35 --L 22 \
    --rollout_steps "$ROLLOUT" \
    --out_dir "$OUT_DIR"

echo "Rsync back to local recommended once done:"
echo "  rsync -av mesu:$OUT_DIR/ \"/Users/marco/Documents/PhD/thermalizer/correction_eval_ks_nu0p35/\""

source deactivate
