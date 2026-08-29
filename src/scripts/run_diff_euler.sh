#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J euler_gamma1.4_diff
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

# Same train+valid chunks as run_emul_euler.sh -- fetched onto $STORE via
# fetch_train_chunks_euler_mesu.sh, staged to $SCRATCH here (idempotent).
# mkdir -p first: rsync's own implicit directory creation only creates the
# final path component, not the whole missing chain (bit us once already).
mkdir -p $SCRATCH/data/euler_multi_quadrants_openBC/data/train $SCRATCH/data/euler_multi_quadrants_openBC/data/valid
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/train/ $SCRATCH/data/euler_multi_quadrants_openBC/data/train/
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/valid/ $SCRATCH/data/euler_multi_quadrants_openBC/data/valid/

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_diffusion_euler.py
source deactivate
