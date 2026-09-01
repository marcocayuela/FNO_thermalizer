#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=01:00:00
#SBATCH -J grid_independence_eval
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

# Usage: sbatch submit_grid_independence_eval.sh [fno_run] [unet_run] [out_dir]
# Short-horizon accuracy only (seq_length-step, no long rollout) at several
# resolutions -- much cheaper than a correction_eval.py rollout job, hence
# the short --time.

module purge
module load python/3.11

source activate fto

mkdir -p /scratch/cayuelam/logs/kolmogorov/

export DATA_DIR=$SCRATCH/data/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

FNO_RUN=${1:-$SCRATCH/fno/runs/kolmogorov/Re90/16modes_emul_RE90}
UNET_RUN=${2:-$SCRATCH/fno/runs/kolmogorov/Re90/unet_delta_RE90}
OUT_DIR=${3:-$SCRATCH/grid_independence_results}

python evaluation/grid_independence_eval.py \
    --data_dir "$DATA_DIR" \
    --exp_dir kolmogorov/Re90 \
    --fno_run "$FNO_RUN" \
    --unet_run "$UNET_RUN" \
    --resolutions 32 64 128 \
    --out_dir "$OUT_DIR"

echo "Rsync back to local recommended once done:"
echo "  rsync -av mesu:$OUT_DIR/ \"/Users/marco/Documents/PhD/thermalizer/$(basename $OUT_DIR)/\""

source deactivate
