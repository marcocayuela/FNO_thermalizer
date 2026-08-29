#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=04:00:00
#SBATCH -J correction_eval
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11

source activate fto

# ── chemins a adapter selon l'emplacement reel sur mesu ──────────────────────
# data/ et runs_mesu/ sont exclus de la synchro (sync-to-mesu.sh respecte le
# .gitignore) : ils doivent deja exister a ces emplacements sur le cluster.
# Pas de rsync ici : les sim*.h5 sont deja presents sur $SCRATCH/data/kolmogorov/Re90/train_traj/
# (utilises par les entrainements precedents), pas besoin de les re-synchroniser pour l'eval.
mkdir -p /scratch/cayuelam/logs/kolmogorov/

export DATA_DIR=$SCRATCH/data/
export RUNS_DIR=${RUNS_DIR:-$SCRATCH/fno/runs/kolmogorov/Re90}
# fno_state vient de WNO_arch (run emul_seq_fno_k1) : code du repo (WNOARCH_DIR, pour
# importer models.model_factory) et checkpoints (WNOARCH_RUNS_DIR) sont a des emplacements distincts.
export WNOARCH_DIR=${WNOARCH_DIR:-$HOME/WNO_arch}
export WNOARCH_RUNS_DIR=${WNOARCH_RUNS_DIR:-$SCRATCH/wno_arch/runs/kolmogorov/Re90}

ROLLOUT=${1:-2000}
SIM_FILE=${2:-sim8.h5}
OUT_DIR=${3:-$SCRATCH/correction_eval_results}
FNO_STATE_RUN=${4:-emul_seq_fno_k1}
# Defauts alignes sur scale_separation.ipynb (diffusion16modes_RE90) plutot que
# les defauts 7/3 de correction_eval.py -- cf. session precedente : s_init=7/3
# ne declenche quasiment jamais la correction sur un rollout court, masquant la
# divergence reelle des emulateurs non stabilises en rollout.
S_INIT=${S_INIT:-1}
S_STOP=${S_STOP:-0}
SEED=${SEED:-101}
# Arrete le rollout d'une trajectoire des que son energie depasse ce facteur x
# l'energie initiale (apres correction eventuelle) -- evite de perdre du temps
# a corriger des milliers de pas sur une trajectoire deja irrecuperable
# (ex: wno_delta diverge des le pas 4 sans ca). 0 desactive.
DIVERGENCE_FACTOR=${DIVERGENCE_FACTOR:-100}
# SKIP_K1_DEFAULTS=1 : n'evalue pas fno_delta/wno_delta/wno_state/fno_state
# (deja en cache, cf. ../results/correction_eval_results_cluster) -- evite de tout
# relancer quand on ajoute juste de nouveaux runs via WNOARCH_SEQ_RUNS.
SKIP_K1_DEFAULTS=${SKIP_K1_DEFAULTS:-0}
# Liste (separee par des espaces) de runs WNO_arch supplementaires a evaluer,
# ex: "emul_seq_fno_k5 emul_seq_wno_k5 emul_seq_fno_k5_delta emul_seq_wno_k5_delta"
WNOARCH_SEQ_RUNS=${WNOARCH_SEQ_RUNS:-}

echo "DATA_DIR         : $DATA_DIR"
echo "RUNS_DIR         : $RUNS_DIR"
echo "WNOARCH_DIR      : $WNOARCH_DIR"
echo "WNOARCH_RUNS_DIR : $WNOARCH_RUNS_DIR"
echo "FNO_STATE_RUN    : $FNO_STATE_RUN"
echo "ROLLOUT          : $ROLLOUT"
echo "S_INIT / S_STOP  : $S_INIT / $S_STOP"
echo "SEED             : $SEED"
echo "DIVERGENCE_FACTOR: $DIVERGENCE_FACTOR"
echo "SKIP_K1_DEFAULTS : $SKIP_K1_DEFAULTS"
echo "WNOARCH_SEQ_RUNS : $WNOARCH_SEQ_RUNS"
echo "Node             : $(hostname)"
echo "GPU              : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="$PYTHONPATH:$(pwd)"

EXTRA_ARGS=()
if [ "$SKIP_K1_DEFAULTS" = "1" ]; then
    EXTRA_ARGS+=(--skip_k1_defaults)
fi
if [ -n "$WNOARCH_SEQ_RUNS" ]; then
    EXTRA_ARGS+=(--wnoarch_seq_runs $WNOARCH_SEQ_RUNS)
fi

python evaluation/correction_eval.py \
    --data_dir         "$DATA_DIR" \
    --runs_dir         "$RUNS_DIR" \
    --wnoarch_dir      "$WNOARCH_DIR" \
    --wnoarch_runs_dir "$WNOARCH_RUNS_DIR" \
    --fno_state_run    "$FNO_STATE_RUN" \
    --rollout          "$ROLLOUT" \
    --sim_file         "$SIM_FILE" \
    --out_dir          "$OUT_DIR" \
    --s_init           "$S_INIT" \
    --s_stop           "$S_STOP" \
    --seed             "$SEED" \
    --divergence_factor "$DIVERGENCE_FACTOR" \
    "${EXTRA_ARGS[@]}"

conda deactivate
