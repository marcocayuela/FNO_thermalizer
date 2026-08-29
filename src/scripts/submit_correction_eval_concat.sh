#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=04:00:00
#SBATCH -J correction_eval_concat
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

mkdir -p /scratch/cayuelam/logs/kolmogorov/

export DATA_DIR=$SCRATCH/data/
export RUNS_DIR=${RUNS_DIR:-$SCRATCH/fno/runs/kolmogorov/Re90}
# Corrector Re-conditionne (cf. fno/fno_2D_classifier_concat.py) : vit sous
# kolmogorov_parametric/, pas sous RUNS_DIR (kolmogorov/Re90/) -- chemin complet
# requis, remplace <RUNS_DIR>/<DEFAULT_DIFFUSION_RUN> via --diffusion_run_dir.
DIFFUSION_RUN_DIR=${DIFFUSION_RUN_DIR:-$SCRATCH/fno/runs/kolmogorov_parametric/diffusion_concat_re50-90_2re}
# Reynolds de la trajectoire evaluee, transmis au corrector -- doit faire partie
# de la grille d'entrainement du run ci-dessus (50 ou 90 pour diffusion_concat_re50-90_2re).
RE=${RE:-90}

ROLLOUT=${1:-5000}
SIM_FILE=${2:-sim8.h5}
OUT_DIR=${3:-$SCRATCH/correction_eval_concat_results}
S_INIT=${S_INIT:-1}
S_STOP=${S_STOP:-0}
SEED=${SEED:-101}
DIVERGENCE_FACTOR=${DIVERGENCE_FACTOR:-100}
# fno_state vient d'un depot externe (WNO_arch) non necessaire pour ce test --
# desactive par defaut, mettre SKIP_FNO_STATE=0 pour l'inclure (et alors
# renseigner WNOARCH_DIR/WNOARCH_RUNS_DIR comme dans submit_correction_eval.sh).
SKIP_FNO_STATE=${SKIP_FNO_STATE:-1}

echo "DATA_DIR          : $DATA_DIR"
echo "RUNS_DIR           : $RUNS_DIR"
echo "DIFFUSION_RUN_DIR  : $DIFFUSION_RUN_DIR"
echo "RE                 : $RE"
echo "ROLLOUT            : $ROLLOUT"
echo "S_INIT / S_STOP    : $S_INIT / $S_STOP"
echo "SEED               : $SEED"
echo "DIVERGENCE_FACTOR  : $DIVERGENCE_FACTOR"
echo "SKIP_FNO_STATE     : $SKIP_FNO_STATE"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="$PYTHONPATH:$(pwd)"

EXTRA_ARGS=()
if [ "$SKIP_FNO_STATE" = "1" ]; then
    EXTRA_ARGS+=(--skip_fno_state)
fi

python evaluation/correction_eval.py \
    --data_dir          "$DATA_DIR" \
    --runs_dir           "$RUNS_DIR" \
    --diffusion_run_dir "$DIFFUSION_RUN_DIR" \
    --re                "$RE" \
    --rollout           "$ROLLOUT" \
    --sim_file          "$SIM_FILE" \
    --out_dir           "$OUT_DIR" \
    --s_init            "$S_INIT" \
    --s_stop            "$S_STOP" \
    --seed              "$SEED" \
    --divergence_factor "$DIVERGENCE_FACTOR" \
    "${EXTRA_ARGS[@]}"

conda deactivate
