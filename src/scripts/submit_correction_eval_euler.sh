#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=08:00:00
#SBATCH -J correction_eval_euler
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

# Usage: sbatch submit_correction_eval_euler.sh [rollout_steps] [traj_idx] [out_dir] [emul_exp_name] [diff_exp_name]
# Needs run_diff_euler.sh and the relevant run_emul_euler*.sh already completed.
# emul_exp_name defaults to the frame_skip=1/state baseline (fno_euler_gamma1.4)
# -- pass fno_euler_gamma1.4_delta / _dt4 / _wide to evaluate a variant instead
# (cf. config_command_emul_euler_delta.yaml/_dt4.yaml/_wide.yaml's own exp_name).
# --time bumped from the original 04:00:00: a rollout well past 99 steps
# (delta/dt4's whole point -- cf. correction_eval_euler.py's --rollout_steps
# help) takes proportionally longer than the original 99-step baseline eval.

module purge
module load python/3.11

source activate fto

mkdir -p /scratch/cayuelam/logs/euler/

export DATA_DIR=$SCRATCH/data/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

ROLLOUT=${1:-99}
TRAJ_IDX=${2:-5}
OUT_DIR=${3:-$SCRATCH/correction_eval_euler_gamma1.4}
EMUL_EXP_NAME=${4:-fno_euler_gamma1.4}
DIFF_EXP_NAME=${5:-diffusion_euler_gamma1.4}

python evaluation/correction_eval_euler.py \
    --data_dir "$DATA_DIR" \
    --val_test_file euler_multi_quadrants_openBC/data/valid/euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5 \
    --traj_idx "$TRAJ_IDX" \
    --emulator_run "$SCRATCH/fno/runs/euler_multi_quadrants_openBC/$EMUL_EXP_NAME" \
    --diffusion_run "$SCRATCH/fno/runs/euler_multi_quadrants_openBC/$DIFF_EXP_NAME" \
    --rollout_steps "$ROLLOUT" \
    --out_dir "$OUT_DIR"

echo "Rsync back to local recommended once done:"
echo "  rsync -av mesu:$OUT_DIR/ \"/Users/marco/Documents/PhD/thermalizer/$(basename $OUT_DIR)/\""

source deactivate
