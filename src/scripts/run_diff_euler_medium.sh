#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J euler_gamma1.4_diff_medium
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

# Same train+valid chunks as run_diff_euler.sh -- fetched onto $STORE via
# fetch_train_chunks_euler_mesu.sh, staged to $SCRATCH here (idempotent).
# mkdir -p first: rsync's own implicit directory creation only creates the
# final path component, not the whole missing chain (bit us once already).
mkdir -p $SCRATCH/data/euler_multi_quadrants_openBC/data/train $SCRATCH/data/euler_multi_quadrants_openBC/data/valid
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/train/ $SCRATCH/data/euler_multi_quadrants_openBC/data/train/
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/valid/ $SCRATCH/data/euler_multi_quadrants_openBC/data/valid/

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# ~3.9M params (~3.9x config_command_diff_euler_small.yaml, ~0.36x the
# full-size run) + noise_sampling_coeff retuned to 0.03 so training actually
# concentrates on the small corruptions this corrector sees in practice.
# Own exp_name (diffusion_euler_gamma1.4_medium), runs alongside the other
# two jobs without touching their files.
python entrypoints/main_diffusion_euler.py configs/config_command_diff_euler_medium.yaml
source deactivate
