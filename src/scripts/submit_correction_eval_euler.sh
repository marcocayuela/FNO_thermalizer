#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=04:00:00
#SBATCH -J correction_eval_euler
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

# Usage: sbatch submit_correction_eval_euler.sh [rollout_steps] [traj_idx] [out_dir]
# Needs both run_emul_euler.sh and run_diff_euler.sh already completed.

module purge
module load python/3.11

source activate fto

mkdir -p /scratch/cayuelam/logs/euler/

export DATA_DIR=$SCRATCH/data/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

ROLLOUT=${1:-99}
TRAJ_IDX=${2:-5}
OUT_DIR=${3:-$SCRATCH/correction_eval_euler_gamma1.4}

python evaluation/correction_eval_euler.py \
    --data_dir "$DATA_DIR" \
    --val_test_file euler_multi_quadrants_openBC/data/valid/euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5 \
    --traj_idx "$TRAJ_IDX" \
    --emulator_run "$SCRATCH/fno/runs/euler_multi_quadrants_openBC/fno_euler_gamma1.4" \
    --diffusion_run "$SCRATCH/fno/runs/euler_multi_quadrants_openBC/diffusion_euler_gamma1.4" \
    --rollout_steps "$ROLLOUT" \
    --out_dir "$OUT_DIR"

echo "Rsync back to local recommended once done:"
echo "  rsync -av mesu:$OUT_DIR/ \"/Users/marco/Documents/PhD/thermalizer/correction_eval_euler_gamma1.4/\""

source deactivate
