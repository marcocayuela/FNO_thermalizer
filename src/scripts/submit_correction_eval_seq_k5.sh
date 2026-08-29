#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=04:00:00
#SBATCH -J correction_eval_seq_k5
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto

mkdir -p /scratch/cayuelam/logs/kolmogorov/

# Pas de rsync : les sim*.h5 sont deja presents sur $SCRATCH/data/kolmogorov/Re90/train_traj/

export DATA_DIR=$SCRATCH/data/
export RUNS_DIR=${RUNS_DIR:-$SCRATCH/fno/runs/kolmogorov/Re90}
export WNOARCH_DIR=${WNOARCH_DIR:-$HOME/WNO_arch}
export WNOARCH_RUNS_DIR=${WNOARCH_RUNS_DIR:-$SCRATCH/wno_arch/runs/kolmogorov/Re90}

ROLLOUT=${1:-2000}
SIM_FILE=${2:-sim8.h5}
OUT_DIR=${3:-$SCRATCH/correction_eval_seq_k5_results}
FNO_SEQ_K5_RUN=${4:-emul_seq_fno_k5}
WNO_SEQ_K5_RUN=${5:-emul_seq_wno_k5}
# Memes defauts que submit_correction_eval.sh/submit_correction_eval_unet.sh --
# alignes sur scale_separation.ipynb (diffusion16modes_RE90) plutot que les
# defauts 7/3 de correction_eval_seq_k5.py.
S_INIT=${S_INIT:-1}
S_STOP=${S_STOP:-0}
SEED=${SEED:-101}
DIVERGENCE_FACTOR=${DIVERGENCE_FACTOR:-100}

echo "DATA_DIR         : $DATA_DIR"
echo "RUNS_DIR         : $RUNS_DIR"
echo "WNOARCH_DIR      : $WNOARCH_DIR"
echo "WNOARCH_RUNS_DIR : $WNOARCH_RUNS_DIR"
echo "FNO_SEQ_K5_RUN   : $FNO_SEQ_K5_RUN"
echo "WNO_SEQ_K5_RUN   : $WNO_SEQ_K5_RUN"
echo "ROLLOUT          : $ROLLOUT"
echo "S_INIT / S_STOP  : $S_INIT / $S_STOP"
echo "SEED             : $SEED"
echo "DIVERGENCE_FACTOR: $DIVERGENCE_FACTOR"
echo "Node             : $(hostname)"
echo "GPU              : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python evaluation/correction_eval_seq_k5.py \
    --data_dir         "$DATA_DIR" \
    --runs_dir         "$RUNS_DIR" \
    --wnoarch_dir      "$WNOARCH_DIR" \
    --wnoarch_runs_dir "$WNOARCH_RUNS_DIR" \
    --fno_seq_k5_run   "$FNO_SEQ_K5_RUN" \
    --wno_seq_k5_run   "$WNO_SEQ_K5_RUN" \
    --rollout          "$ROLLOUT" \
    --sim_file         "$SIM_FILE" \
    --out_dir          "$OUT_DIR" \
    --s_init           "$S_INIT" \
    --s_stop           "$S_STOP" \
    --seed             "$SEED" \
    --divergence_factor "$DIVERGENCE_FACTOR"

conda deactivate
