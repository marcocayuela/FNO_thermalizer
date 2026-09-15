#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=1:00:00
#SBATCH -J shear_flow_eval_native
#SBATCH -o /scratch/cayuelam/logs/shear_flow/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/shear_flow/ $SCRATCH/data/shear_flow

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Rollout-based generalization check on the 4 held-out test_traj initial
# conditions -- see evaluation/correction_eval_shear_flow.py's own docstring
# for why this is a stronger test than the Tr/Te split reported during
# training. --rollout 190: each test trajectory has 200 frames.
python evaluation/correction_eval_shear_flow.py \
    --exp_dir shear_flow/Re5e4_Sc1e0_native \
    --run_name exp_shear_flow_Re5e4_Sc1e0_native \
    --rollout 190 \
    --out_dir $SCRATCH/correction_eval_shear_flow_native_results \
    --device cuda
source deactivate
